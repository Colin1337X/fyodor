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

`assemble.py` reproduces that exact PTX hash and invokes CUDA 13.1 ptxas
offline for SM120 at O3. The baseline uses 64 physical registers per thread;
the metadata candidate uses 66. Both report zero stack frame and zero spill
stores/loads. Thus fewer PTX metadata instructions do not establish a lower
resource cost: the candidate increases register use. `assembly.json` records
the compiler identity and source/tool/log hashes. This is not runtime driver-JIT
allocation, measured occupancy or speed; raw parity and timing still decide
whether to keep the candidate. No CUDA context was created by this check.

Raw GPU parity, memory checking and synthetic timing have now completed below.
Real-model correctness and accepted decode timing remain required for integration.
Production dispatch is unchanged and removing this optional source leaves the
production build unchanged. GPU checks ran after the contiguous-readback
sanitizer completed successfully. Accepted decode scores remain 85.61025294% short and
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
end-to-end timing.

## Completed raw validation and diagnostic

The raw sweep and Compute Sanitizer each passed all 207 cases; the sanitizer
exited zero with exactly one zero-error summary. The eight-shape probe completed
with byte-identical timed outputs, forward/reverse ordering and ten samples per
kernel/shape. Mean candidate latency was 1.09%-3.50% below the native baseline.
For the dominant 3840-column/15360-row expansion, means were 0.124483872 ms native
and 0.122826559 ms candidate (1.33% lower). For 15360 columns/3840 rows they were
0.140362432 and 0.138830976 ms (1.09% lower).

These repeated synthetic weight-bank measurements have no continuous activity
audit and are not accepted decode results. The modest gain warrants isolated
real-model validation, not production admission. Preserve existing precision
tolerances and compare against the same input-reuse/readback baseline.

## Isolated real-model integration

`build_model.py` verifies the raw sanitizer and completed readback baseline,
then creates a separate CUDA override and four C17/C23 executables. It appends
the metadata source to the existing NVRTC module and changes only the selected
four-token Q4 kernel. The FFN input reuse, contiguous readback, lifecycle harness
and production archive are preserved. A diagnostic selection marker plus
positive execution counts prevents silent fallback from passing this check.

`run_lifecycle.py c17|c23 257|8192` retains the validated reset/rewind and logit
checks and requires the metadata marker. Both C17 and C23 at prefix 257 pass
both reset paths with maximum scaled error 2.19546455e-5, 568 integer calls,
and one contiguous download per four-row verification. Each result's 101 hashes
independently verified. The C17 8192-prefix lifecycle also passed both reset
paths with maximum scaled error 1.63989979e-5, 568 integer calls and single-download
assertions. Its 101 hashes verified after completion. `run_generation.py` requires these lifecycle proofs before testing
whole-model output parity. No accepted end-to-end speedup or production dispatch
is introduced by this isolated integration.

`run_model_memcheck.py` is prepared for the isolated model binary and requires
completed lifecycle and generation proofs before launching Compute Sanitizer.
Its validator additionally rejects a missing metadata-kernel selection marker;
eight CPU-only evidence tests pass. It has not run on the GPU yet.

`run_monitored.py` targets the isolated metadata binary and requires successful
model sanitizer, lifecycle and generation proofs before GPU execution. It keeps
the existing process/idle checks, continuous activity audit and input hashes,
and additionally requires the metadata selection marker. Three input-integrity
tests pass. This runner is prepared but has not executed a timing workload.

Short-context generation now passes under both C17 and C23: twelve matching
ordinary/serial2/batch2/joint3 cases per build, forward/reverse/forward order,
25 prompt tokens and 32 generated tokens. Each joint round recorded 2556
integer calls, the metadata selection marker was present, and each result's
111 hashes verified after completion. Long-context C17 generation is running;
its output and timing are not yet accepted. Full-model sanitizer and monitored
performance validation remain outstanding for this candidate.

## Offline launch-bound follow-up

`audit_launch_bound.py` compiles an additional entry from the exact metadata
source, changing only its symbol and minimum blocks-per-SM launch bound from
three to four. In the same SM120 offline module, ptxas reports 64 registers for
the baseline, 66 for the current metadata kernel and 61 for the bound-four
variant, all with zero stack frame and zero spills. The generated source hash,
transformation, tool identity and input hashes are recorded in
`launch-bound-assembly.json`. This does not establish runtime JIT resources,
occupancy, parity or speed. It justifies testing this launch bound after the
current GPU run finishes. No running input or production dispatch was changed.
