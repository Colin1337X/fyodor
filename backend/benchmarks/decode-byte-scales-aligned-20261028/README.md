# 160-byte Q4 metadata layout diagnostic

Original Fyodor implementation, base eef98b2. Diagnostic only in root optcpp;
no runtime, model-file or build changes. Reference is the accepted integer
scale-product gate kernel at the same fixed width. Candidate retains ordered
arithmetic, F32 activations and exact original Q4 coefficients, changing only
packed weight layout. No upstream source copied.

The first4 bytes retain half d/minimum. Bytes4..11 contain unpacked six-bit
scales and12..19 unpacked six-bit minima. Payload starts at32, occupying128
bytes. Twelve padding bytes give a 160-byte block and 32-byte aligned payload. Metadata loads four words and extracts bytes.
Storage grows11.11% versus144-byte native Q4 blocks. The160-byte
experiment changes both alignment and metadata loading, so it does not isolate
their individual effects. Both variants use64 registers/thread on the tested
model-sized cases. Historical192-byte float metadata was rejected separately.

## Validation and measurement

Six shapes with random metadata/nibbles match byte-for-byte including guards.
Every bank in two rotating cases is checked. Six additional cases cycle14 finite half bit patterns spanning signed zero, subnormal, normal and maximum finite values; all match.
Conversion expands metadata losslessly on CPU before upload; no precision mode
or reduction order changes. NaN/Inf model coefficients and real-model runtime
execution are not covered here.

RTX5060Ti SM120, driver617.42, NVRTC compute_120 C++11 fmad=true and
NYA_CUDA_FAST_MATH=1. Ten samples/variant, five in each opposite order,500 graph
kernels/sample after warmups. Hot cases reuse weights; rotating uses eight banks.
Raw weight_bank_bytes counts native bytes; summary.json explicitly includes the
larger candidate footprint. Both representations coexist for diagnostics; this
is not a runtime peak-memory or preparation-cost measurement. Packing/upload
costs are excluded from kernel timing. All5 monitored activities pass idle
and competitor gates; no competitor killed and no unrelated build/GPU work
overlaps timing. CPU preprocessing finishes before each measured case.

| Case | Columns x rows | Reference ms | Candidate ms | Throughput gain |
|---|---:|---:|---:|---:|
| hot | 256 x 9 | 0.001103 | 0.001070 | +3.08% |
| hot | 2048 x 5632 | 0.038041 | 0.029090 | +30.77% |
| hot | 4096 x 11008 | 0.161422 | 0.145563 | +10.89% |
| extended | 768 x 65 | 0.001687 | 0.001633 | +3.33% |
| extended | 2304 x 257 | 0.006293 | 0.005430 | +15.90% |
| extended | 8192 x 28672 | 0.797689 | 0.796604 | +0.14% |
| rotating | 2048 x 5632 | 0.041207 | 0.036524 | +12.82% |
| rotating | 4096 x 11008 | 0.166086 | 0.146933 | +13.04% |

## Decision

Proceed to a width2048 runtime trial with C-owned repacking/cache lifetime and
native fallback. Require allocation/size checks, real-model CPU correctness,
C17/C23, physical-removal checks and matched decode timing before acceptance.
The wider8192 hot case has essentially no gain, so do not enable all widths.

Current whole-model scores remain102.59% short/121.41% long versus llama.cpp;
these diagnostics do not update them. >=125% on both remains unmet. The updated
ungated, graph-disabled profile in decode-byte-scales-20261028 reports gate/up
361.112ms of913.425ms matrix work, supporting further work on that operation;
profiling includes submission gaps and is not acceptance timing.

Reproduce run.py serially with fresh output directories; aligned variant also
has run_scales.py. Source manifests and raw samples are retained. No runtime
integration, packing amortization, memory-budget handling, AMD/Vulkan or
training gain is claimed.
