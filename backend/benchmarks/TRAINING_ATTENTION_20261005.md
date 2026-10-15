# Resident causal/local training attention — October 5, 2026

The private CUDA training device now supports causal and local grouped-query
attention forward/backward, including future visibility within nonzero local
group IDs. It saves normalization statistics rather than a square probability
matrix and recomputes probabilities in bounded query tiles during backward.
A complete real TinyLlama frozen attention branch matches CPU autograd through
Q/K/V projections, RoPE, attention, output projection and two microbatches.
Public CLI/UI training remains on CPU; graph ownership, loss and transactional
optimizer integration are still required for full GPU training.

## Memory and numerical contract

The plain C descriptor names dense F32 Q/K/V matrices, query/KV head counts,
head dimension, positive finite scale, local window and an optional U32 group
buffer. Q/K/V share sequence length. Query head count must be a multiple of KV
head count. With window zero the mask is causal. With a positive window, past keys
start at `max(0,row-window+1)`; matching nonzero group IDs additionally permit
future keys, exactly as in the CPU training graph. Group IDs need not be
contiguous or small. Empty/malformed views, arithmetic overflow, buffer capacity
errors, stale/foreign/opaque handles and destination aliasing fail before launch.

Forward saves two doubles per token/query head: maximum rounded score and
normalization mass, totaling `16*N*H` bytes. The first pass calculates the maximum,
the second sums F32-rounded exponentials in double precision, and the last forms
F32 probabilities and accumulates values in key order with double intermediates.
Score products and dot sums use double precision before CPU-compatible F32 score
rounding. Invalid or out-of-F32-range scores produce nonfinite output for the
caller's explicit status check. The range check happens **before** the cast,
including values slightly beyond FLT_MAX that could round back to finite FLT_MAX.

Backward uses caller-owned workspace of `16*N*H*min(N,16)` bytes. Each tile has
at most 16 query rows, storing probability and score derivative as double pairs
(the probability itself is F32-rounded). The prepare kernel reconstructs
probabilities from unchanged Q/K and saved statistics, computes `dP=dY*V`, and
applies the softmax derivative. Q-gradient writers visit keys in order;
K/V-gradient writers visit query rows and their associated query heads in order.
Tiles execute in increasing row order on the same stream. Contributions round
to F32 and accumulate into existing gradients, without floating-point atomics.
Either or both of Q/K gradients may be omitted; V-only backward avoids the dP
work. At least one gradient is required, and gradient destinations must be
separate from each other and all inputs/state/workspace.

Forward is one kernel. Backward uses one prepare kernel per tile, one Q kernel
when requested, and one combined K/V kernel when either is requested. Dispatch
adds no allocation, transfer or host fence. Saved inputs/groups/state must remain
unchanged until backward, with the same descriptor; numerical checks and step
acceptance remain explicit.
A device failure poisons the context; numerical failure is recoverable and must
reject the step before an optimizer commit. No CPU replay is added.

Saved state and workspace are linear in N once N >= 16. For N=2,048 and H=32,
state is 1 MiB and backward workspace is 16 MiB, versus 512 MiB for one saved
F32 probability matrix. This is size arithmetic, not a measured 2,048-token
full-model training run. This first implementation recomputes dots in multiple
passes and uses ordered reductions; it is not claimed to match a tensor-core
FlashAttention implementation or establish training throughput superiority.

## Source study

Original Fyodor C/CUDA code. The source study used MIT-licensed llama.cpp/ggml
revision `5266f24da75dc449bd56cbed7addb9c8e4a6a73e`, tracing probability
recomputation and backward equations in
[ggml-cpu/ops.cpp](https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/ggml/src/ggml-cpu/ops.cpp)
and block reductions/bounded score processing in
[fattn-vec.cuh](https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/ggml/src/ggml-cuda/fattn-vec.cuh).
Fyodor retains its own F32 probability rounding and ordered gradient semantics,
not ggml tensors, its mixed-precision storage or its online-softmax arithmetic.
No upstream source was copied. The snapshot records revision and license/source
hashes alongside compiler, GPU, model and executable identities.

## Correctness and real-model composition

Primitive comparisons use `1e-6*(1+abs(reference))`, without relaxing the
existing primitive bound. They check every output and two backwards into
nonzero gradients across 96 ordinary causal/local shape cases: 1/3/15/16/17/33/65/257
tokens, dimensions 1/2/7/32/64/257, and both grouped-query layouts. Additional
cases exercise all six partial-gradient combinations, window-one attention,
groups ignored by global causal attention, noncontiguous UINT32_MAX group IDs,
and 1,025-token attention. Independent double finite differences check every
Q/K/V element in small causal and local/grouped problems. State/workspace/output
tail sentinels and malformed/aliased descriptors cover storage boundaries.

One hundred queued scope reuses add exactly 1,000 kernels without transfers or
host fences and produce bit-identical ordered V-gradient accumulation. Seven
injected launch failures cover forward plus every backward kernel in two query
tiles. Overflow checks include positive/negative scores that round back to
FLT_MAX, and recovery on the same device after rejecting the numerical result.

The real-model modes `attention8` / `attention64` use TinyLlama Q4_K_M's actual
first-layer frozen Q/K/V/output matrices, 32 query heads, four KV heads and
64-wide heads. The native CPU graph uses six matrix workers and the same
inputs/weights. Every output and accumulated input-gradient element is checked
after each of two warmups and five measured repetitions, with an unchanged
scaled tolerance of `1e-5`. Each repetition is two complete microbatches.

| Tokens | Output error | Input-gradient error | State bytes | Workspace bytes | Peak arena bytes |
|---:|---:|---:|---:|---:|---:|
| 8 | 5.0837521e-7 | 6.0285853e-7 | 4,096 | 32,768 | 6,185,472 |
| 64 | 5.0837521e-7 | 4.553005e-7 | 32,768 | 524,288 | 11,637,248 |

Each repetition queues 64 kernels at eight tokens or 82 at 64 tokens, with no
uploads/downloads inside execution and one explicit completion fence. Setup
performs seven uploads; 21 downloads check status/output/gradient after seven
repetitions. The eight-token reference-math mode and the earlier Q/K-only probe
also pass. Raw timings remain in the evidence but are excluded from performance
acceptance because interactive desktop activity was not isolated. The probes
have frozen projection weights and no optimizer/loss/convergence trajectory;
they are not a completed trainer.

## Validation and instrumentation limits

All 15 matrix stages passed: C17 CPU (53 applicable tests), C17 and C23
CUDA/Vulkan (112 each), C23 ONNX (54), ASan/UBSan (53), physical removal of
optcpp (53), CUDA (53), and CUTLASS (107), plus 11 frontend tests and the
production frontend build. Each backend configuration reports four unavailable
ROCm/MLX hardware tests as skipped.

The final complete primitive suite passed CUDA memcheck, initcheck, synccheck
and racecheck. Every tool reports zero errors, and racecheck reports zero
hazards/warnings. Final records include the exact test executable hash.
Racecheck uses `--force-synchronization-limit 4 --racecheck-num-workers 2` to
bound instrumentation pressure; it checks every launch, without a filter,
skip count or truncated launch window. Its additional fences are tool behavior,
not production dispatch behavior. Uninstrumented tests separately assert no
new host fences/transfers and correct queued scratch reuse.

Two earlier default Racecheck runs exited with Windows `0xc0000409` in
`ucrtbase.dll`, without a hazard report. Those are failed runs, not evidence of
correctness. The traced run reached the final 1,025-token case, and its target's
working set was observed above 6 GiB during larger shapes. These observations
suggest instrumentation resource pressure but do not prove the internal abort
cause. The bounded full run and subsequent final full sanitizer suite passed.
The original outputs, traced failure, Windows event fields and bounded run are
retained. Wallpaper Engine was stopped under the existing authorization before
the bounded runs; other applications were preserved. No performance conclusion
is drawn from any sanitizer run.

Reproduce from the worktree root using the configurations recorded with the
preceding RoPE milestone and bundled w64devkit on PATH:

```powershell
python backend/benchmarks/training-20260923/validate.py --output <fresh-matrix-directory>
python backend/benchmarks/training-attention-20261005/run_probes.py
python backend/benchmarks/training-attention-20261005/sanitize.py --output <fresh-sanitizer-directory>
pwsh -NoProfile -File backend/benchmarks/training-attention-20261005/package.ps1
python backend/benchmarks/training-attention-20261005/verify_package.py
python backend/benchmarks/training-attention-20261005/snapshot.py
```

Probe/sanitizer scripts refuse to overwrite run logs. Use a fresh checkout or
fresh evidence location when repeating them. Package build logs append attempt
numbers. The first focused build tried the new target before CMake regenerated
it; the next compilation caught a signed conversion in a test counter. Both
logs are retained, and both setup issues were corrected before validation.

The NSIS package was built and verified on October 6. Extracted binaries and
runtime libraries match the supplied build artifacts (with only Tauri's expected
bundle marker change in the desktop executable). The packaged trainer completed
the CLI lifecycle test; eight HTTP checks covered authentication, model load,
native CUDA selection, repeatable 725-token-prompt/eight-token generation,
runtime inspection and unload. Optional BLAS/CUTLASS were disabled for that
generation check and no CPU replay occurred. The package was extracted and
exercised, not installed or interactively UI-tested. Identity and responses are
recorded in `desktop-package.json`.

The embedded inference module is byte-identical to the preceding worktree
milestone (32,843 bytes, SHA-256
`5d307a7bafd596eeb2f8fb886383db315edbc679562664e89bf627c56594ea4d`).
Training kernels remain in the separate training module. This change does not
claim an inference speedup, complete GPU training or loss/optimizer support.
