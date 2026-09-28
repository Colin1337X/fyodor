# Resident RMSNorm forward and backward — September 29, 2026

The private CUDA training device now supports weighted and unweighted RMSNorm,
with accumulating input and normalization-scale gradients. Composed with the
resident FFN, it matches native CPU autograd using real TinyLlama weights. Public
training remains on CPU while embeddings, attention, loss, graph ownership and
transactional optimizer integration are unfinished.

## Contract and implementation

`nya_train_device_rms_norm` accepts a dense F32 matrix, an optional F32 weight
vector and a positive finite F32 epsilon. It overwrites the output and writes
one double inverse RMS per row into a caller-owned buffer. That buffer needs
`rows * 8` bytes and must survive until backward. Input, weight and saved inverse
must remain unchanged between forward and backward; the private API does not
track arbitrary writes to reconstruct graph dependencies.

One 256-thread block owns each row. Threads accumulate squares in double and
use a fixed shared-memory reduction tree. Double accumulation preserves the
CPU precision contract for extreme finite F32 inputs whose squares overflow F32.
Backward input gradients use a similar double reduction of weighted gradient
products. A separate one-writer-per-column kernel accumulates scale gradients
in row order, rounding each contribution to F32 as CPU autograd does. It reuses
the saved inverses instead of repeating the square reduction for each column.

Forward launches one kernel. Backward launches one kernel per requested
gradient, at most two, and performs no allocation, transfer or fence. Gradients
are optional independently; a scale gradient requires a weight input. Input and
scale gradients may share a destination only for a single row, in which case
stream order adds the input contribution before the weight contribution for
each element. Other output/input aliases are rejected. Read-only inputs may
share storage. All handles, capacities, shapes, size arithmetic and grid bounds
are validated before dispatch. Numerical checking remains explicit. A failure
between backward kernels poisons the device; it cannot expose a partial result
as a completed operation.

This is original Fyodor code. Source study examined llama.cpp/ggml's MIT-licensed
[norm.cu](https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/ggml/src/ggml-cuda/norm.cu)
and `norm.cuh` at `5266f24da75dc449bd56cbed7addb9c8e4a6a73e`. Its per-row block
reductions informed the work decomposition. Fyodor retains its own double
arithmetic, saved state, scale derivatives and accumulation semantics; no
upstream source or C++ tensor machinery was copied. Source/license hashes,
compiler, model, GPU and base revision are in the
[snapshot](training-norm-20260929/snapshot.json).

## Correctness

[Six focused tests](training-norm-20260929/initial-tests.log) passed: CPU-only
wrapper behavior, normal/reference GPU execution, and injected failures in the
forward, input-gradient and scale-gradient launches. The fixture covers:

- Thirty shapes (1/3/33 rows and ten widths from 1 through 2,048), each with and
  without weights, compared to CPU autograd at `1e-6 * (1 + abs(reference))`.
- Repeated gradient accumulation, optional gradients, shared input/weight
  storage and a shared single-row gradient destination.
- Central finite differences for all input and scale elements of a small case.
- Zero-valued rows, subnormal rows and alternating signed maximum finite F32 values,
  with the smallest positive F32 epsilon.
- Output, gradient and saved-inverse tail guards; no hidden allocation,
  transfer or synchronization; malformed, overflowed, undersized, stale,
  foreign and aliased descriptors rejected before changing output.

The backward scale test compares accumulation through two CPU graphs. The
small independent double objective checks derivatives separately from the CPU
autograd formulas. Hardware fault tests verify failure propagation through later
dispatch, read and finish calls, including failure after input gradients have
already been queued.

## Real normalized FFN

The extended TinyLlama Q4_K_M probe uses first-layer gate/up/down matrices and
the model's actual FFN normalization scale and epsilon. Projection weights stay
compressed and frozen; the normalization scale has a trainable gradient. CPU
autograd uses six workers and compares every final output, accumulated input
gradient and accumulated scale gradient.

Each repetition executes two identical microbatches with fresh scratch scopes,
retaining weights, inputs, output, gradients and numerical status. Fifty-six
launches include temporary-buffer initialization and selected finite checks.
There are no in-repetition host transfers and one final completion fence. Six
setup uploads and four final downloads cover the entire probe. The 64 MiB arena
includes caller-owned saved inverse state; reported usage excludes CUDA context
and module overhead.

| Tokens | Maximum scaled output error | Input-gradient error | Scale-gradient error | Peak arena bytes | Retained bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.52560806e-7 | 3.24168447e-7 | 8.72230728e-7 | 24,289,600 | 22,716,672 |
| 64 | 3.13362458e-7 | 5.93153494e-7 | 2.35406709e-6 | 37,135,104 | 24,551,680 |

The unchanged composed-probe bound is
`max(abs(GPU - CPU) / (1 + abs(CPU))) <= 1e-5`. Two warmups precede five recorded
repetitions; each resets gradients and numerical status. Final outputs, gradients
and status are inspected after the final repetition. Earlier repetitions check
dispatch/completion errors but do not individually download numerical results.
The probe has no attention, loss, optimizer, checkpoint or convergence claim.
Timings are preserved but excluded from performance acceptance because desktop
activity was not isolated.

Reproduce with `NYA_CUDA_BLAS=0`, `NYA_CUDA_CUTLASS=0` and `NYA_COMPUTE=cpu`:

```powershell
build-cuda/nya-bench-training-device.exe .tools/tinyllama-q4_k_m.gguf rmsffn8
build-cuda/nya-bench-training-device.exe .tools/tinyllama-q4_k_m.gguf rmsffn64
python backend/benchmarks/training-norm-20260929/sanitize.py
```

CPU model loading is intentional; the harness explicitly creates a CUDA training
device. Previous `ffn8`/`ffn64` modes remain available without normalization.
The [unnormalized eight-token regression probe](training-norm-20260929/legacy-ffn8.out)
retains exactly the previous output/gradient maximum errors, arena usage and
transfer counts. Its timing is likewise excluded from performance acceptance.
Inference's generated CUDA bundle is unchanged at 32,423 bytes, SHA-256
`927af8f2929259b9984b8b82ddf4a327f8194a4d9603947d4758765613822d55`.

## Configuration and package validation

The complete RMSNorm fixture passed all four GPU sanitizers: memcheck, initcheck,
synccheck and racecheck. Each process exited zero with no reported errors or
hazards; no launch window was excluded. [Commands and exits](training-norm-20260929/sanitizers.json).

All [15 configuration stages](training-norm-20260929/matrix/validation.json)
passed. CPU C17 and ASan/UBSan each passed 50 applicable tests, CUDA/Vulkan C17
and C23 each passed 91, and ONNX passed 51. Physical-removal configurations passed
50 without optcpp, 50 without CUDA and 86 without CUTLASS. Each configuration
skipped four unavailable ROCm/MLX hardware checks. All 11 frontend tests and the
production frontend build passed.

The rebuilt desktop installer is 406,271,143 bytes, SHA-256
`524160944d419994d44dcefa6d1ec16fc65e03b63dfea1c9c48052f83f4895c2`.
[Package verification](training-norm-20260929/desktop-package.json) matches
extracted backend resources to the built binaries, checks the desktop bundle
marker, passes the extracted trainer CLI lifecycle and eight loopback HTTP
requests. Native CUDA inference with optional matrix libraries disabled produces
the same eight-token continuation twice from a 725-token prompt without CPU
replay. No installation or interactive UI test is claimed.

[The final audit](training-norm-20260929/audit.json) verifies source hashes,
all 15 matrix stages, four complete GPU sanitizer runs, both normalized FFN
results, the prior unnormalized probe's correctness/memory counters, inference
bundle identity and package evidence. These results do not establish complete
GPU training or accepted throughput improvement.
