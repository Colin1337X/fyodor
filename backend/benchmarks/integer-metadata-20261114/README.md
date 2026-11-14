# Native Q4 metadata unpacking experiment

Original Fyodor candidate in `optcpp/matvec/integer_metadata.cu`. The earlier
160-byte packed layout was correct but 23%-34% slower in synthetic diagnostics.
This candidate keeps native 144-byte Q4_K blocks, three aligned metadata word
loads, the existing float4 input, eight-row/bound-three launch shape, and the
same weight expression, floating-point accumulation and warp reduction order.
It reconstructs four scale/minimum bytes together with integer masks/shifts
before extracting individual groups. It allocates no new weight copy.

`check_metadata.py` checks the host algebra against an independent byte decoder:
all 65536 byte pairs for high scale/minimum reconstruction and 10000 random
12-byte blocks pass. This establishes an algebra identity only, not execution
of the CUDA kernel.

`compile_only.py` compiles with NVRTC compute_120, C++11, fmad=true and the
existing FAST_MATH setting, without loading the CUDA driver or creating a GPU
context. `inspect_ptx.py` compiles the baseline and candidate together. Static
PTX and.b32 sites fall 24 to 19, or.b32 8 to 2, and shr.u32 22 to 16. Both have
three metadata word-load sites, four quantized-byte load sites, eight float4
load sites, 32 FMA sites and 20 shuffle sites. Static PTX is not SASS, dynamic
instruction count, physical register pressure, occupancy or elapsed time.

No GPU execution, raw parity, memory checking or timing has occurred for this
candidate. Those checks remain required before choosing or integrating it.
Production dispatch is unchanged and removing this optional source leaves the
production build unchanged. The active contiguous-readback sanitizer retains
exclusive use of the GPU. Accepted decode scores remain 85.61025294% short and
106.8707255% at 8K versus llama.cpp; the requested 125% on both is not achieved.

## Prepared GPU validation

`check_kernel.py` compares the candidate and native eight-row/bound-three kernel
against the existing Q4 oracle across 207 shapes/coefficient-edge cases, with
guarded interleaved inputs/outputs and independent float64 bounds on small cases.
It requires the completed readback sanitizer, validates compile/algebra proof
hashes and checks for another GPU workload before importing the context-creating
module. Its current prerequisite refusal was exercised without GPU execution.

`run_memcheck.py` repeats the same sweep under Compute Sanitizer, preserving
terminal evidence and requiring zero errors plus complete parity. `probe.py`
requires both raw and sanitizer evidence before comparing native/metadata kernels
on eight shapes in forward/reverse order. It includes the same interleave call
in each path, warmup and ten event observations per kernel/shape. This remains
a synthetic diagnostic without continuous contamination coverage, not accepted
end-to-end timing. All four harnesses parsed; none has executed GPU work yet.
