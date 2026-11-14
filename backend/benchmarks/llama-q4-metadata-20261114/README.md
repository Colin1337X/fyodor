# llama.cpp decode implementation adaptation

Pinned reference: ggml-org/llama.cpp revision
`5266f24da75dc449bd56cbed7addb9c8e4a6a73e` (b10809).
The inspected decode chain is `llama_decode` -> `llama_context::decode` ->
`process_ubatch` -> `graph_compute` -> backend graph execution. CUDA selects
quantized matrix-vector operations in `ggml-cuda.cu`; `mmvq.cu` dispatches
`mul_mat_vec_q` and its Q4_K specialization to `vec_dot_q4_K_q8_1` in
`ggml/src/ggml-cuda/vecdotq.cuh`. That path quantizes activations to Q8_1,
uses packed dot products, architecture/batch-specific launch choices and
optional gate fusion. Those choices cannot all be copied into Fyodor's F32
activation path without changing its numerical contract.

Source links:
- https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/src/llama-context.cpp
- https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/ggml/src/ggml-cuda/mmvq.cu
- https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/ggml/src/ggml-cuda/vecdotq.cuh

## Implemented adaptation

`optcpp/matvec/llama_q4_metadata.cu` directly adapts the two-group, 16-bit
scale/minimum reconstruction from `vec_dot_q4_K_q8_1`. The surrounding
four-token projection is Fyodor's native-layout implementation. It retains
144-byte Q4_K blocks, F32 float4 activations, the integer scale-times-quant
product, and the original per-output accumulation and warp reduction order.
It does not embed ggml or import foreign tensor objects. The MIT copyright
notice is in the source and the complete license is preserved beside it as
`optcpp/matvec/LICENSE.llama-cpp`.

The previous native metadata candidate reconstructed four groups in 32-bit
words. This candidate instead follows upstream's two-group reconstruction and
loads six 16-bit metadata words. The different load/register tradeoff needs
measurement; upstream's use of this code in a Q8 dot product does not prove
that it will be faster in Fyodor's F32 projection.

## Evidence and limits

`check_decoder.py` extracts the actual adapted decoder statements, compiles
them at O3 with strict warnings under C17/C23, and checks against an independent
byte decoder. Both pass 65536 exhaustive byte pairs plus 10000 independent
metadata blocks. This is host metadata validation, not CUDA projection parity.

`compile_only.py` builds the CUDA source through NVRTC compute_120 without a
GPU context. `inspect_ptx.py` compiles baseline and candidate together;
`assemble.py` verifies that exact PTX hash and assembles offline for SM120.
The candidate uses 70 registers/thread versus 64 for the baseline; neither
reports a stack frame or spills. Runtime driver JIT and measured performance
may differ. No GPU execution or performance measurement has occurred yet.

Required next gates: guarded raw output parity, Compute Sanitizer, real-model
parity and clean paired decode timing. The running readback sanitizer has
exclusive GPU use. This candidate has no production dispatch or build dependency;
physically removing it leaves the production build unchanged. It must not be
enabled on the basis of host tests or compiler evidence alone.

Accepted decode scores remain 85.61025294% short and 106.8707255% at 8K
relative to llama.cpp. Neither score is changed by this experiment.

## Prepared GPU checks

`check_kernel.py` compares both the adapted kernel and native eight-row kernel
to the existing projection oracle in 207 cases, including 196 finite-half
coefficient combinations, output/input guards and independent float64 bounds
for the small shapes. It verifies the compilation and actual-host-decoder
evidence before loading the CUDA module. The pending-readback-sanitizer refusal
was exercised successfully; no GPU work was started.

`run_memcheck.py` repeats that sweep under Compute Sanitizer and requires a
successful process exit, exactly one zero-error summary, the exact ordered
shape/coefficient inventory, parity/guards and small-shape float64 checks.
Six CPU-only `test_memcheck.py` tests pass, including rejection of duplicate
cases, invalid/missing/duplicate summaries and missing/NaN/failed bounds.
`probe.py` requires raw and sanitizer evidence before measuring eight shapes
in forward/reverse order. These harnesses are prepared, not GPU-validated.
