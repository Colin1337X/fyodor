# TurboQuant C cache and 512-wide portable attention

Original code, explicit opt-in. Usage/algorithm/storage/limitations are in
../../docs/turboquant.md and ../../../optcpp/attention/README.md.
TurboQuant is a CPU text-inference precision mode, not a GPU speed acceptance.
FlashAttention-style tiled online softmax now accepts head widths through 512
on the existing CUDA and Vulkan providers. Upstream FA4 AOT is still pending.

## Validation

- C17: seven codec/inference/KV/attention tests passed, including Vulkan 257/512
  widths and unchanged CPU attention oracles. Final codec/inference rerun passes.
- C23: eleven selected tests passed, including FDPP short, graph prefix replay,
  Vulkan and CPU attention. Final codec/inference rerun passes.
- Clang AddressSanitizer + UBSan: both TurboQuant tests pass, including final
  synthetic shared-KV and sliding-window replay cases.
- Physical non-C removal: 63 tests passed, four expected ROCm/MLX hardware skips.
  optcpp, CUDA/cutlass and CUDA/cute physically absent from a copied source tree.
  Native C codec/inference tests remain present. Tiny synthetic removal timings
  are build smoke checks, not performance acceptance.
- CUDA portable kernel: nine independent double-oracle cases, including 257/512
  widths, causal batches, GQA, windows, tail tiles and extreme logits. Maximum
  scaled error <8e-8, unchanged 1.5e-4 threshold. Memcheck, racecheck, initcheck
  and synccheck all return zero errors/hazards.
- Real Gemma 12B portable F32 attention: prefixes 7/8/9 pass the unchanged 0.001
  scaled CPU-logit tolerance; largest observed scaled error 0.000334831487.
  This is a short correctness check, not a performance or long-context claim.

Codec tests cover widths 32/63/128/512, 2/3/4 bits, packed tails, byte accounting,
zero vectors, invalid dimensions/capacity, finite-input rejection, atomic failed
stores, GQA/window attention versus independently evaluated dequantized KV,
and seed-ensemble QJL behavior. QJL mean 0.995536892 versus expected 1,
variance 0.0126761555 over 256 independent seeds. These do not establish
quality neutrality for any actual language model.

## Real-model lossy precision diagnostics

Twelve deterministic input tokens, native CPU F32 baseline, separate 3-/4-bit
sessions. Reset and single-token replay exactly reproduce each mode's logits.
Models are the local TinyLlama Q4_K_M and requested Gemma4 12B Q4_K_M.

| Model | 3-bit logit RMSE / max absolute | 4-bit logit RMSE / max absolute |
|---|---:|---:|
| TinyLlama | 1.64707 / 9.00109 | 0.685858 / 3.58553 |
| Gemma4 12B | 3.90450 / 20.13572 | 1.72867 / 8.62641 |

These differences are substantial. No F32 parity, perplexity/retrieval-quality
acceptance, or speedup is claimed. Long-context model quality and GPU-resident
TurboQuant remain outstanding. The mode is off by default and is clearly
identified in benchmark JSON. No accepted F32 score is replaced by these runs.

## Reproduce

Build backend with C17/C23; run `ctest -R 'turboquant|attention-|kv-precision'`.
`nya-test-turboquant-inference MODEL.gguf` runs the real-model diagnostic.
`python check_kernel.py NEW_OUTPUT_DIRECTORY` uses local CUDA/NVRTC and runs the
direct device oracle; wrap with compute-sanitizer --tool TOOL --error-exitcode 9
for each of memcheck/racecheck/initcheck/synccheck. Use
`NYA_CUDA_PORTABLE_ATTENTION=1 NYA_COMPUTE=cuda nya-test-inference MODEL resident 9`
for the real-model F32 oracle. `verify-without-non-c` creates a separate source
copy and builds/tests it. Ordinary builds require neither Python nor a C++
compiler for TurboQuant. The portable shader is checked in as SPIR-V.

Initial compile diagnostics are retained: fixed a local error-buffer size name
and made the nonzero norm comparison explicit for Clang's strict warnings.
Existing numeric thresholds were not relaxed.
