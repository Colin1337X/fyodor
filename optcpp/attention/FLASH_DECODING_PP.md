# Flash-Decoding++ attention, original Fyodor implementation

The immediate implementation priority is Flash-Decoding++ before FA4.
flash_decoding_pp.cu implements the unified-scale split attention technique
from section 3 of Hong et al., https://arxiv.org/html/2311.01282v3 . No code or
binary from the author-linked, unlicensed wheel artifact is incorporated.
This is an attention implementation, not a copy of the full engine. Flat GEMM
double buffering and hardware-adaptive matrix dispatch are still outstanding.

Configure NYA_ENABLE_OPTCPP=ON and NYA_ENABLE_PORTABLE_ATTENTION=ON, then set
NYA_CUDA_FLASH_DECODING_PP=1. Default execution stays unchanged. The flag takes
precedence over portable online attention for a single decoded query with
head width <=256. Prefill, reference arithmetic and unsupported widths retain
existing execution. Root optcpp can be physically absent; no host C++ compiler
or optional library is required. The two kernels are runtime-compiled by NVRTC.

Each head has one or four independent KV partitions, all using phi=0. Each
partition writes its unnormalized numerator, denominator and retry flag into
request-owned, budget-accounted scratch. The combine launch sums numerators
and denominators. There are no float atomics or cross-block spin waits. GPU
graphs retain their existing two-context scheduling policy; request metadata
supplies the current position on replay. Windowing and GQA are preserved.

A score outside [-16,16], including NaN, or a value with magnitude >1e12 marks
the partition for retry. These conservative limits protect exponent evaluation
and accumulation for uint32 context lengths. The whole head is then recomputed
with stable online softmax by the combine kernel, without host synchronization.
No clipping, silent downcast, tolerance weakening or probability truncation is
used. Nonfinite input is not promised a finite output; normal runtime rejection
semantics still apply. The stable path retains ordinary F32 arithmetic limits.

No tensor cores, warp-size assumptions, TMA or TMEM are used. Compilation has
been checked for compute 75/80/86/89/90/100/120, execution only on SM120.
HIP syntax portability is not validated AMD support. Vulkan currently has the
separate online attention shader; porting this split unified-scale path and
validating actual RDNA/CDNA devices remain required work.

Validation and raw evidence: backend/benchmarks/flash-decoding-pp-20261016.
The independent double oracle uses a 2e-6 scaled-error limit and covers short,
long, odd widths, GQA, windows, empty partitions and forced stable retries.
All four CUDA sanitizers passed. No performance improvement is claimed from
these correctness tests; accepted scores remain 93.18% short / 114.31% long.

The first monitored end-to-end test regressed 15.87% short and 64.29% long
against the existing native path. It remains opt-in; it is not an accepted
performance optimization. A CUDA-only coalesced specialization in
flash_decoding_pp_warp.cu is currently diagnostic and not loaded by the runtime.
