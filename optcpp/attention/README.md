# Portable attention integration (in progress)

Status: original experimental F32 tiled-online-softmax CUDA/Vulkan kernels and
an original Flash-Decoding++ unified-scale decode attention implementation are
integrated. This is not the complete FlashDecoding++ engine (its flat GEMM and
hardware-specific pipelines remain pending), or completed FlashAttention-4 support.
RDNA/CDNA execution and hardware validation remain outstanding. The default
runtime stays on its previously validated kernels. No performance gain claimed.

## Audited sources

Pinned revisions and file hashes are in upstream-audit.json. FA4's official
SM120 forward kernel subclasses its SM80 implementation, uses SM80-era MMA,
and limits shared memory to 99 KB. It accepts FP16/BF16, not our F32 KV contract.
The pinned interface rejects split-KV and paged KV on SM120. Thus claiming
that every FA4 kernel is incompatible with SM120 would be incorrect, but the
B200 pipeline cannot simply be used on this GPU or on AMD.

FA4 is CuTeDSL, with a Python/PyTorch-facing interface. NVIDIA's pinned CuTeDSL
example demonstrates AOT object + C-header export. A real FA4 bridge must use
that export route behind a Fyodor-owned C ABI, with generated objects and all
non-C dependencies isolated here. The ordinary runtime must not import Python
or expose foreign tensors. Half-precision kernels need a separate explicit
precision contract; never silently downcast the accepted F32 benchmark.

The author-linked Flash-Decoding++ artifact provides a Linux CPython 3.10
inficom wheel and model drivers. Its tree exposes no CUDA/C++ source or license
file. No artifact code/binary has been copied or installed. The paper's
unified-maximum softmax, flat GEMM and scheduling techniques require an
original implementation or a separately license-verified source. The portable
portable_f32 kernel uses stable online softmax. flash_decoding_pp.cu implements
the paper's section 3 unified-scale partitioned attention with guarded
whole-head stable recomputation. See FLASH_DECODING_PP.md for its contract.

## Pipeline design and remaining work

Keep public descriptors and dispatch/ownership in C. Separate capabilities by
provider, precision, head dimensions, query count, mask/window, KV layout,
shared-memory limits and device architecture. A rejected capability selects
native fallback before scheduling; a runtime execution error remains an error.

The common compatibility path uses tiled QK, online maximum/denominator and
streaming V accumulation. No full score matrix, tensor-core instructions,
TMEM, TMA or warp-width assumptions occur in portable_f32.cu. CUDA/HIP source
syntax does not itself prove HIP compatibility: actual HIP compilation,
ROCm launch integration and RDNA/CDNA numerical runs are required next.

Specialized CUDA pipelines may subsequently add cp.async or MMA where supported.
AMD needs independently selected wave32/wave64/LDS pipelines and native HIP
module/stream ownership. Capability selection must not equate RDNA and CDNA,
or infer support simply because a device is called Blackwell. FA4 AOT kernels
remain a distinct optional precision-specific provider. The Flash-Decoding++ CUDA attention path now provides
bounded-exponent detection and stable recomputation; its Vulkan/ROCm ports remain
pending. No fixed statistical maximum may silently overflow.

## Current opt-in and removal

Configure NYA_ENABLE_PORTABLE_ATTENTION=ON with CUDA enabled, then set
NYA_CUDA_PORTABLE_ATTENTION=1 at runtime. The initial kernel accepts head widths
up to 256, GQA, causal batched queries, sliding windows and device request
positions. Other widths and reference execution retain existing kernels.
NYA_ENABLE_OPTCPP=OFF or a physically absent root optcpp directory removes all
references to this kernel. No host C++ compiler is enabled. CUDA's existing
runtime-compiled kernels remain in their independently removable compartment.

Build target verify-without-non-c copies source without optcpp, CUDA/cutlass
or CUDA/cute, configures with optional attention requested, builds and runs
regressions. It never renames/deletes the working checkout. Separate existing
CUDA/CUTLASS/optcpp removal targets remain supported. Build and test evidence
is under backend/benchmarks/portable-attention-20261016.

## Validation to date

Six direct numerical cases cover widths 64/80/128/256, GQA, context 0-1024,
batched causal queries, sliding windows including width one, tail tiles,
request-driven position, and +/-100 inputs producing large logits. A double
precision independent reference uses stable softmax. Maximum scaled error
is below 8.1e-8 with the existing 1.5e-4 limit. All four CUDA sanitizers pass.
The real TinyLlama CPU-logits oracle passes unchanged tolerance. This is not
proof of exact CPU/GPU KV equality or complete model/backend coverage.

Removal validation passes 157 tests with five expected provider skips.
Flash-Decoding++ C17 (39/39) and C23 (40/40) regression and graph lifecycle
checks pass after fixing scratch-driven graph selection.
Pending: controlled latency/memory timing, AMD compilation/runtime, AOT FA4
integration, Vulkan Flash-Decoding++ and flat-GEMM/dataflow specialization.
The existing 93.18% short / 114.31% long scores are not scores for this kernel.

Vulkan: NYA_VULKAN_PORTABLE_ATTENTION=1 selects the original online-softmax
shader, not Flash-Decoding++. Existing double-oracle tests pass at 2e-6 and the
TinyLlama full inference test passes. The current Vulkan adapter uploads host
KV for each call; resident KV and performance optimization remain pending.
