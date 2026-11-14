# Native Q4 batch-four gate/up fusion: offline candidate

The current decode profile identifies Q4 batch-four FFN expansion as the largest
matrix group. CUDA/resident.inc fuses Gemma gate/up only at count one; batched
execution uses separate gate/up projections followed by nya_vector GELU.

llama.cpp revision 5266f24da75dc449bd56cbed7addb9c8e4a6a73e was inspected locally:
ggml/src/ggml-cuda/mmvq.cu supports gate fusion and applies the GLU operation
after reducing both projections. Its MIT license was checked. This is an
algorithm reference, not a source transplant: q4_batch_gate.cu derives from
Fyodor's q4_pair_rows.cu and uses the existing nya_vector GELU expression.
No upstream implementation text was copied. Unlike llama.cpp's Q8 activation
dot-product route, this candidate retains interleaved F32 inputs and native
144-byte Q4_K blocks; it introduces no activation quantization contract.

One warp computes the same row from gate and up matrices, sharing each float4
input load across both projections. Per-projection accumulation and warp
reduction order follow the existing kernel. The output is the row-major gated
result for all four inputs. A future dispatch must require compatible Q4_K
gate/up shapes and strides, batch four, and the existing input layout; other
cases must retain fallback. No runtime dispatch or allocation changes exist.

NVRTC compilation and offline SM120 assembly pass. ptxas reports 80 registers,
no stack or spills, and no barriers for the candidate. These are static resource
reports, not measured occupancy or a speedup. No CUDA context was created.
The full-model sanitizer for the separate joint-one scheduling experiment
remains the only active GPU workload.

Required next evidence: compare against two existing native projections plus
the exact GELU operation, independent numerical bounds, guarded partial rows,
shape rejection, sanitizer checks, paired synthetic timing, then full-model
parity and monitored performance before considering integration. Fusion can
regress due to scheduling or activation latency; compilation alone establishes
neither correctness nor performance. Accepted scores remain unchanged.

## Guarded numerical harness prepared

check_kernel.py prepares 222 cases plus eight unsupported-shape checks. Gate and
up use independently generated native weights. Each separate projection must
meet the existing independent C float64 oracle bound; fused output must then be
byte-exact against those projections followed by the existing nya_vector GELU.
All final values must be finite. Inputs, both weight arrays, interleaved-input
guards and output guards are checked. Cases cover warp/block row boundaries,
the full 15360-by-3840 shape, finite half-coefficient extremes and zero, constant,
alternating, subnormal, large and impulse activations. Large activations are
bounded at 2^12 here to keep the gated product finite; this is not a claim of
all-input overflow handling.

The combined reference/candidate harness compiles offline, with all required
entry points present. preflight-refusal.json records actual refusal while the
joint-one full-model sanitizer prerequisite is missing, before importing the
context-creating CUDA helper. These are preparation checks, not numerical passes.
After that sanitizer completes successfully, run check_kernel.py without flags
for raw validation. Sanitizer labels only select separate evidence filenames;
they do not themselves invoke or certify Compute Sanitizer.
