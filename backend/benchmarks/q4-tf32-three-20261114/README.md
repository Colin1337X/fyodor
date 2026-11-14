# Four-token native Q4 tensor-core experiment

The refreshed short profile attributes roughly 131-137 ms per joint case to
Q4 batch-four expansion alone. Launch-bound, partial-unroll and shared-input
variants did not provide a reliable replacement. This original experiment
changes the matrix algorithm rather than adding another launch-bound variant.

`optcpp/matvec/q4_tf32_three.cu` dequantizes native 144-byte Q4_K blocks directly
into register fragments, reads the existing interleaved four-token F32 input,
and uses three TF32 products (high/high and both high/residual cross terms)
with F32 accumulation. It allocates no expanded weight bank. The padded eight
output columns retain only the four actual tokens. Each warp handles 16 rows;
tail rows supply zero fragments while all lanes participate in each MMA.
The launch contract is a one-dimensional grid and a (128,1,1) block on SM80+.

Fragment coordinates follow NVIDIA's primary PTX ISA description:
https://docs.nvidia.com/cuda/archive/11.5.2/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-1688
Fyodor already uses a CUTLASS three-product TF32 operator for optional prefill;
this new decode experiment has no CUTLASS or foreign runtime dependency.
No upstream source is copied; PTX instruction syntax and documented register
coordinates are used by an original kernel.

This changes accumulation order and omits the residual/residual product.
It is not claimed to be exact F32 arithmetic. Admission must retain the
existing raw independent error bounds and the 0.0005 full-model scaled-logit
bound. Finite extreme values, cancellation, subnormals, TF32 conversion
overflow, tail guards and reset/rewind behavior still need GPU validation.
It must not be dispatched merely because compilation succeeds. A production
path would also require explicit unsupported-input fallback and architecture
checks; neither is implemented here.

`compile_only.py` compiles compute_120 PTX with NVRTC 13.1 and records source,
compiler, log and PTX hashes without loading the CUDA driver. `assemble.py`
verifies that evidence and invokes ptxas offline for SM120. These establish
compilation only. Raw numerical checks, sanitizers, real-model correctness,
performance and runtime integration are pending. No production source,
archive or dispatch changes; deleting this optional file leaves builds intact.
Accepted llama.cpp scores remain 85.61025294% short / 106.8707255% at 8K.

SM120 offline assembly reports 40 registers, no stack frame or spills, and
no barriers for the prototype. These are compiler resource reports, not
achieved occupancy or a measured speedup. Compilation and assembly evidence
hashes independently verified before committing. GPU execution is pending.

The independent numerical oracle is native C with double coefficients,
products, sums and absolute-product sums. C17/C23 shared libraries compiled
at O3 with strict warnings; each passed 198 finite fixtures against Python
`math.fsum`, three nonfinite-coefficient rejections and an invalid-argument
check. It has no CUDA dependency and uses none of the device metadata helpers.

`check_kernel.py` is prepared for 219 cases: 17 shapes including row-tile
tails, all 196 finite-half coefficient pairs at 17 rows, and six additional
bounded input patterns (zero, one, alternating, subnormal, large, impulse).
Both the native oracle kernel and tensor-core candidate must satisfy the
existing `2e-6 * sum(abs(double products)) + 1e-6` bound on every output,
including the large shapes. Input/weight/output guards and independent
interleave layout checks remain required. Eight unsupported-shape calls use
null inputs and must leave output untouched. Finite extremes near F32 overflow
are not covered by these bounded patterns and remain an admission limitation.

The harness verifies compilation, assembly and host-oracle proof hashes and
requires the active QKV full-model sanitizer to finish before creating a GPU
context. Its pending-prerequisite refusal was exercised. None of these 219
GPU cases has run yet; preparation and host oracle checks are not candidate
GPU correctness evidence. No numerical tolerance was relaxed.
