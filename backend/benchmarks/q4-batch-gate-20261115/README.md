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
