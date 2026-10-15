# Resident rotary training operations — October 5, 2026

The private CUDA training device now runs RoPE forward and backward with no
intermediate allocation, transfer or synchronization. Adjacent-pair LLaMA and
split-half Gemma layouts match the CPU graph. Testing also found and corrected
large-angle trigonometric error in the MinGW CPU training path. Public CLI/UI
training still executes on CPU; this does not complete resident attention,
loss, optimizer, graph ownership or checkpoint integration.

## Contract and implementation

The source is a dense F32 matrix with one token per row and `heads*dimension`
columns. Dimension must be positive and even. A resident F32 frequency buffer
contains `dimension/2` angular frequencies shared by all heads. Position is the
zero-based row number, matching the current CPU graph. Zero frequencies express
unrotated pairs; arbitrary position offsets, frequency gradients and YaRN are
not added by this interface.

One GPU thread owns a pair. It computes the angle, sine/cosine and products in
double precision, writes F32 forward values, or adds the F32 inverse rotation
to an existing input gradient. No floating-point atomic operations or saved
activations are needed. Backward deliberately accumulates rather than overwrites.
The kernel preserves both supported pair layouts including head boundaries.

The C dispatcher checks handles, shape arithmetic, dimensions, capacities, layout
flags and output/input aliasing before launch. Read-only source/frequency aliases
are permitted. Opaque token-index maps cannot masquerade as buffers. Invalid
descriptors are recoverable; driver/launch errors poison the context through the
existing device contract. Frequencies must remain unchanged between forward and
backward. Device-value validation remains an explicit finite-check operation;
queue success is not proof of numerical validity or permission to commit an
optimizer step.

This is original Fyodor C/CUDA code. The source study traced MIT-licensed
llama.cpp/ggml [rope.cu](https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/ggml/src/ggml-cuda/rope.cu)
and `rope.cuh` at `5266f24da75dc449bd56cbed7addb9c8e4a6a73e`: adjacent and
split-half pairing, and backward through the inverse rotation. Fyodor retains
its own descriptors, explicit frequencies, double intermediates and accumulating
gradients. No upstream source or tensor machinery was copied. License/source
hashes are recorded in [snapshot.json](training-rope-20261005/snapshot.json).

## CPU discrepancy and correction

The first GPU suites passed ordinary frequencies and 8,193-token sequences,
then failed at row 1 with `frequency=FLT_MAX`. CPU output was `1.12973797321`,
while CUDA produced `-1.34228027`. Both normal and reference CUDA modes failed
the same comparison; the [initial tests](training-rope-20261005/focused-tests.log)
and [localized diagnostic](training-rope-20261005/diagnostic-test.log) are retained.

A standalone compiled probe showed MinGW's static `cos`/`sin` returning
`-0.2139451813683188` / `0.976845668142761` for that angle. Calling the
already-linked Windows CRT imports returned `0.8530210398303042` /
`-0.5218765233336585`, agreeing with CUDA and an independent mpmath evaluation at
100 decimal digits. Separate calls and `long double` did not fix the static
routines. [Probe source](training-rope-20261005/trig_probe.c), raw outputs and
[oracle generation](training-rope-20261005/oracle.py) preserve the investigation.

CPU training forward/backward now use the existing CRT imports on MinGW when
`abs(angle) >= 65536`. Ordinary angles and other platforms retain standard C
math. This introduces no new library, dynamic-loader state, dependency or C++
code. A hardcoded high-precision forward/backward regression runs even in
CPU-only builds, avoiding the broken host functions as its own oracle. The
GPU comparison tolerance remains `1e-6*(1+abs(reference))`; it was not relaxed.
This correction is scoped to training RoPE, not an audit of every host-math use.

## Primitive and composition evidence

The primitive suite checks 84 ordinary shape/layout combinations (dimensions
2/6/64/128/256/514/1024, row counts 1/3/65, one or three heads), plus long-position
and extreme-frequency cases for each layout. It covers zero/negative/subnormal
frequencies, finite differences of every input in both small layouts, two
backward accumulations into a nonzero gradient, tail sentinels, malformed sizes,
aliases, stale/foreign/opaque handles, explicit nonfinite status and recovery.
One hundred queued scratch scopes add exactly 300 kernels, no transfers or
fences, and preserve the persistent gradient. Injected failures independently
exercise forward and backward launch poisoning.

The real-model `qkrope8` and `qkrope64` probes use TinyLlama Q4_K_M's actual first
query/key Q4_K matrices (32 query heads, four KV heads, dimension 64, width
2,048), its RoPE base, and deterministic dense activations/gradient seeds. CPU
six-worker autograd supplies every Q/K output and the shared input gradient
from two microbatches. GPU backward follows CPU reverse traversal: key then
query. Every element is checked after each of two warmups and five measured
repetitions, against the unchanged `1e-5` scaled bound.

| Tokens | Maximum Q error | Maximum K error | Input-gradient error | Peak arena bytes |
|---:|---:|---:|---:|---:|
| 8 | 1.80671342e-6 | 1.78329042e-6 | 0 | 3,080,704 |
| 64 | 2.39978066e-6 | 2.39991739e-6 | 0 | 6,062,592 |

Each repetition queues 30 kernels for two microbatches, adds no uploads or
downloads, and has one explicit completion fence. Setup performs six uploads;
28 downloads validate status and all three arrays across seven repetitions.
The eight-token reference-math probe also passes, as does the prior normalized
FFN probe. These are frozen projection components, not attention, parameter
updates, convergence or end-to-end training. Timings remain in the raw JSON but
are **excluded from performance acceptance** because concurrent builds and
interactive desktop activity were not isolated.

## Reproduction and validation

The work was isolated from unrelated UI/server changes in the primary checkout.
`configure.py` records the five local SDK configurations; it uses the supported
standalone backend CMake entry point. The first attempt encountered the broken
system CMake installation; the preserved log precedes the successful explicit
bundled-CMake retry. The first package attempt lacked the copied optional
CUTLASS license beside its DLL; the license was supplied and that failure log
is retained too.

From the worktree root, with bundled w64devkit on PATH:

```powershell
python backend/benchmarks/training-rope-20261005/configure.py
python backend/benchmarks/training-20260923/validate.py --output <fresh-matrix-directory>
python backend/benchmarks/training-rope-20261005/run_probes.py
python backend/benchmarks/training-rope-20261005/sanitize.py --output <fresh-sanitizer-directory>
pwsh -NoProfile -File backend/benchmarks/training-rope-20261005/package.ps1
python backend/benchmarks/training-rope-20261005/verify_package.py
python backend/benchmarks/training-rope-20261005/snapshot.py
```

Probe/output scripts refuse to overwrite existing run evidence. Use a fresh
checkout/evidence destination for a complete rerun. Configuration logs append
attempt numbers. Compiler/model/source identities, raw test logs, sanitizer
logs and installer verification are adjacent to this report.

All 15 validation stages passed: C17 CPU (52 applicable tests), C17 and C23
CUDA/Vulkan (102 each), C23 ONNX (53), ASan/UBSan (52), physical removal of
optcpp (52), CUDA (52), and CUTLASS (97), plus 11 frontend tests and the production
frontend build. Each backend configuration reports four unavailable ROCm/MLX
hardware tests as skipped. All four complete CUDA sanitizer runs report zero
errors; racecheck also reports zero hazards and warnings. No launch window was
excluded from these runs.

The NSIS package built from this isolated checkout was extracted and checked
against its supplied binaries and runtime libraries. Its trainer completed the
CLI lifecycle test; its backend passed eight authenticated/unauthenticated HTTP
checks including repeatable 725-token-prompt, eight-token native-CUDA generation
with optional BLAS/CUTLASS disabled and no CPU replay. This verifies the packaged
binaries, not an interactive installation or UI session. Installer identity and
responses are preserved in `desktop-package.json`.

Inference kernel content remains unchanged: the fresh worktree's CRLF checkout
expanded the raw embedded module from 32,423 to 32,843 bytes, so raw hashes differ.
After LF normalization, the historical and current modules are byte-identical
(SHA-256 `76313ca8c3b0ee61f4b7d6e805a337ca773617cc7021651d0b5d4d140942a680`).
The snapshot records both raw identities and the explicit comparison. No
inference throughput improvement or complete GPU training is claimed.
