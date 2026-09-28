# Resident activations and gated FFN — September 28–29, 2026

The private CUDA training device now supports scale, SiLU, tanh-approximation
GELU, softcap, addition and multiplication, with accumulating backward kernels.
Together with resident matrices and scratch scopes, these operations execute a
complete gated feed-forward branch without intermediate host transfers. Public
CLI/UI training still uses CPU autograd: normalization, embeddings, attention,
loss, graph ownership and transactional optimizer integration remain unfinished.

## Contract

Dense F32 forward operations overwrite their destinations; backward operations
add to initialized gradients. Unary formulas use double intermediates and the
CPU graph's stable negative SiLU branch. Binary views carry independent row and
column dimensions; either dimension broadcasts when it is one. Each gradient
element has one writer, visiting expanded contributions in CPU row-major order.
This avoids nondeterministic floating-point atomics. Large broadcast reductions
are serial per destination element and have not been performance optimized.

Either binary input gradient may be omitted. A shared gradient destination is
supported for equal operand shapes, with both contributions interleaved at each
position to preserve CPU accumulation order. Read-only inputs may share storage.
Destinations cannot overlap inputs. Invalid operations, scalars, dimensions,
overflowed sizes, undersized buffers and stale/foreign handles fail before
dispatch. These calls add no allocation, transfer or synchronization; numerical
checks remain explicit. Driver/launch failure poisons the device.

## Numerical and residency checks

[The new fixture](../tests/test_training_elementwise.c) compares all four unary
operations against CPU autograd and central finite differences at eight lengths
around warp/block boundaries. It covers finite extremes, signed zero, subnormals,
tail guards and repeated accumulation. Both binary operations cover 14 shape
pairs, including scalar, row, column, matrix and outer broadcasting. Optional
gradients, shared-gradient rounding order, invalid descriptors and an injected
launch failure are tested.

A synthetic fully trainable branch computes
`0.75 * (down(silu(gate(X)) * up(X)) + X)` with X[3,5], gate/up[7,5]
and down[5,7]. Across two microbatches, persistent weights/gradients survive
fresh scratch scopes. All outputs and all input/weight gradients match CPU
autograd. The fixture asserts 68 queued launches and no in-graph transfers or
fences. [Nine focused training-device tests](training-elementwise-20260928/composed-tests.log)
passed, including normal/reference elementwise execution and failure injection.

The complete new elementwise fixture, including this composed branch, passed
memcheck, initcheck, synccheck and racecheck with zero reported errors/hazards.
No launch window is excluded. [Commands and exits](training-elementwise-20260928/sanitizers.json).

## Real TinyLlama component

The harness uses TinyLlama Q4_K_M first-layer gate/up/down weights, 2,048 input
features and 5,632 hidden features. A six-worker CPU autograd graph provides every
final output and accumulated input-gradient element as an oracle. Weights remain
frozen; this is a pure SiLU FFN, without normalization, attention, loss or optimizer.

Each repetition runs two identical microbatches, retaining compressed weights,
input, output, gradient seed, accumulated input gradients and numerical status
outside temporary scopes. It queues 40 launches, adds no in-repetition upload or
download, and uses one final completion fence. Two warmups precede five recorded
repetitions. Status and gradients reset per repetition; full numerical comparison
and status inspection apply to the final repetition. Earlier repetitions are
checked for dispatch/completion errors, not individually downloaded for parity.

The composed-probe tolerance was fixed before measurement at
`max(abs(GPU - CPU) / (1 + abs(CPU))) <= 1e-5`. Existing primitive tolerances
were unchanged.

| Tokens | Maximum output error | Maximum accumulated input-gradient error | Peak arena bytes | Retained bytes |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 1.65849492e-6 | 1.70720346e-6 | 24,142,080 | 22,700,036 |
| 64 | 1.65849492e-6 | 2.27811414e-6 | 36,069,632 | 24,535,044 |

Both use a 64 MiB arena, five setup uploads and three final downloads. Arena
counts exclude CUDA context/module overhead. Timings are retained in the
[snapshot](training-elementwise-20260928/snapshot.json) but **excluded from
performance acceptance** because desktop activity was not isolated. This proves
component correctness and bounded reuse, not training throughput or convergence.

Reproduce with `NYA_CUDA_BLAS=0`, `NYA_CUDA_CUTLASS=0` and `NYA_COMPUTE=cpu`
(CPU model loading; the harness explicitly creates its CUDA training device):

```powershell
build-cuda/nya-bench-training-device.exe .tools/tinyllama-q4_k_m.gguf ffn8
build-cuda/nya-bench-training-device.exe .tools/tinyllama-q4_k_m.gguf ffn64
python backend/benchmarks/training-elementwise-20260928/sanitize.py
```

## Provenance and inference isolation

Source study inspected llama.cpp/ggml at
`5266f24da75dc449bd56cbed7addb9c8e4a6a73e`, under its MIT license:
[unary.cu](https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/ggml/src/ggml-cuda/unary.cu)
and `unary.cuh`, including SiLU backward. Its contiguous flattening and explicit
stream dispatch inform the interface. Fyodor's implementation is original,
using its CPU formulas, accumulating derivatives and one-writer broadcast
reductions. No upstream code or tensor/runtime machinery was copied. Exact
source/license hashes, model identity, compiler and hardware are in the snapshot.

The inference CUDA bundle remains byte-identical to the module-split baseline:
32,423 bytes, SHA-256
`927af8f2929259b9984b8b82ddf4a327f8194a4d9603947d4758765613822d55`.
Only the separate training bundle gains these kernels.

## Configuration and package validation

All [15 validation stages](training-elementwise-20260928/matrix/validation.json)
passed. CPU C17 and ASan/UBSan each passed 49 applicable tests; CUDA/Vulkan C17
and C23 each passed 85; ONNX passed 50. Physical-removal configurations passed
49 without optcpp, 49 without CUDA and 80 without CUTLASS. Each skipped four
unavailable ROCm/MLX hardware checks. All 11 frontend tests and the production
frontend build passed.

The rebuilt desktop installer is 406,323,421 bytes, SHA-256
`a5edcc1d72c898b7856d3d64ab14928324750ccf60fba7549a9071ce31f82ef3`.
[Package verification](training-elementwise-20260928/desktop-package.json)
matches extracted backend resources to the current binaries, checks the desktop
bundle marker, passes the extracted trainer CLI lifecycle and eight loopback
HTTP requests. Native CUDA inference with optional matrix libraries disabled
produces the same eight-token continuation twice from a 725-token prompt without
CPU replay. No installation or interactive UI test is claimed.

[The final audit](training-elementwise-20260928/audit.json) verifies source
hashes, all 15 matrix stages, four complete sanitizer runs, both real FFN results,
the unchanged inference bundle and package evidence. Throughput acceptance and
complete GPU training remain explicitly unproven.
