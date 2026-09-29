# Resident embeddings and ordered matrix gradients — September 29, 2026

The private CUDA training device now gathers embeddings and accumulates their
gradients without intermediate host transfers. Token IDs bind once to immutable
resident metadata; repeated IDs accumulate in original token order. A real
TinyLlama embedding → RMSNorm → FFN probe also exposed a matrix-backward
rounding mismatch, which is fixed without changing the numerical tolerance.
Public CLI/UI training remains on CPU. Attention, loss, graph ownership and
transactional optimizer integration are still required for complete GPU training.

## Index ownership and kernels

`nya_train_device_indices` validates host IDs, sorts `(row, position)` pairs in
portable C, and uploads a packed map containing the original IDs, sorted unique
rows, offsets and ordered occurrence positions. For N IDs and G unique rows,
the payload is `(2*N + 2*G + 1) * 4` bytes. Host staging uses O(N) memory and
O(N log N) sorting; it does not allocate in proportion to vocabulary size.
Rows/count must fit U32 and the CUDA grid bound also limits token count.

Binding is an explicit setup boundary: it allocates in the existing bounded
arena, zero-initializes storage and performs one upload with the normal transfer
fences. Later host-ID changes cannot mutate the map. Map handles share the
never-reused device namespace and scratch lifetime, but generic buffer APIs
reject them. They cannot be read, overwritten, zeroed, used as tensor input or
passed as a destination to other kernels. Only the embedding dispatcher accesses
their sealed payload. Invalid/stale/foreign map handles are rejected before a
launch, and scratch retirement invalidates maps created inside that scope.

Forward assigns a block to each token and decodes its selected row directly
from F32, F16, BF16, Q4_0, Q8_0, Q4_K or Q6_K storage into F32 output. Backward
assigns a block to each unique referenced row. Each column has one writer that
adds occurrences to an existing F32 gradient in token order. Untouched rows stay
unchanged; no vocabulary-wide ID scan or floating-point atomics are required.
Each operation dispatches one kernel with no allocation, transfer or fence.

This is original Fyodor C/CUDA code. Source study inspected MIT-licensed
llama.cpp/ggml [getrows.cu](https://github.com/ggml-org/llama.cpp/blob/5266f24da75dc449bd56cbed7addb9c8e4a6a73e/ggml/src/ggml-cuda/getrows.cu)
and `getrows.cuh` at `5266f24da75dc449bd56cbed7addb9c8e4a6a73e`. Its forward
gathers/dequantizes selected rows; its backward scans all IDs for each destination
row. Fyodor instead prepares stable occurrence groups at upload time. No upstream
source or C++ tensor representation was copied. The
[snapshot](training-embedding-20260929/snapshot.json) records source/license hashes,
compiler, GPU, model identity and base revision.

## Numerical failure found by composition

The first eight-token composed probe failed at embedding-gradient element 2,850:
CPU `0.885441780`, GPU `0.885470867`, scaled error `1.5427189e-5` against the
unchanged `1e-5` bound. The CUDA reference mode also failed (`1.39097606e-5`).
Both [normal](training-embedding-20260929/embedffn8.log) and
[reference](training-embedding-20260929/embedffn8-reference.log) failures are retained.

The previous matrix input-gradient kernel reduced products through two F32
chains and added their combined sum to the destination. CPU autograd rounds
each product and adds it in reduction order to the existing gradient, including
gradients already contributed by another branch. The difference became visible
after normalization and repeated-token accumulation. The CUDA input-gradient
kernel now starts with the existing gradient and uses explicit rounded F32
multiplication/addition in CPU order, without fused multiply-add. Forward
inference and training matrix kernels are unchanged.

A new adversarial test confirmed the same issue in the matrix weight-gradient
kernel: adding four ones separately to `2^24` must round away each contribution;
adding their sum changes the result. The
[failing regression](training-embedding-20260929/order-regression-before.log)
was recorded before applying the same correction to weight gradients. The test
also distinguishes a rounded product followed by addition from fused arithmetic.
The [exact kernel diff](training-embedding-20260929/matrix-accumulation-order.patch)
and [source identities](training-embedding-20260929/matrix-order-experiment.json)
preserve this experiment. No tolerance was loosened and CPU training semantics
were not changed. The serial dependency chain may cost throughput; no isolated
performance acceptance is claimed here.

## Correctness and lifetime evidence

The portable index-map fixture checks 80 distributions, including 4,097 tokens,
all-identical IDs and a U32-sized vocabulary, validating ordered unique keys,
offsets, full permutation coverage and stable positions. It also tests malformed
and overflowed requests. CUDA tests cover 112 storage/shape cases: three floating
types across eight widths and four token counts, plus four quantized types across
four token counts. Every gathered value matches the native decoder exactly;
two accumulated backward passes match CPU autograd bit for bit. Tail guards,
host-ID mutation after binding, opacity, wrong/stale/foreign handles, allocation
budget failure and launch failure are covered.

One hundred queued scratch scopes reuse a persistent map and table gradient,
with 300 launches and no added host transfers or synchronization. Separate
repeated-ID and matrix-gradient rounding tests exercise large pre-existing
gradients. [All 21 focused device tests](training-embedding-20260929/final-tests.log)
passed, including the existing matrix, finite-check, elementwise and RMSNorm
fixtures. The first build caught an uninitialized test array in an invalid-input
case under `-Werror`; initializing the array fixed the fixture before execution.
The failed build log remains in the evidence folder.

## Real trainable embedding component

The composed probe decodes TinyLlama's entire 32,000 × 2,048 embedding table to
F32 and treats it and the first-layer normalization scale as trainable parameters.
Gate/up/down weights retain their mapped compressed representations. Token IDs
use `k % 5 ? vocabulary - 1 - k % 3 : 1`, giving four distinct rows and repeated
tokens at both tested lengths. CPU autograd uses six workers. It compares every
final output, all 65,536,000 embedding-gradient entries (including untouched rows)
and all normalization-scale gradients across two identical microbatches.

Each repetition queues 64 launches including temporary initialization and
selected finite checks. It adds no in-repetition upload/download and has one
final completion fence. Seven setup uploads and four final downloads include
the full F32 embedding table and its full gradient. The arena reserves 768 MiB;
reported usage excludes CUDA context/module overhead and CPU oracle storage.

| Tokens | Maximum scaled output error | Embedding-gradient error | Scale-gradient error | Peak arena bytes | Retained bytes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 1.83100899e-7 | 4.55228527e-6 | 8.84322511e-7 | 548,577,856 | 547,004,928 |
| 64 | 1.83100899e-7 | 6.48548272e-6 | 1.45553556e-6 | 561,423,872 | 548,840,448 |

All pass `max(abs(GPU - CPU) / (1 + abs(CPU))) <= 1e-5`. Two warmups precede
five recorded repetitions. Gradients and status reset per repetition, and full
numerical comparison/status inspection occurs after the final repetition.
Earlier repetitions check dispatch/completion errors but are not individually
downloaded for parity. The final reference-mode eight-token probe also passes.
The earlier normalized and unnormalized eight-token FFN modes still pass with
unchanged memory and transfer counts. [Commands and exits](training-embedding-20260929/probes.json).

These are component tests, without attention, loss, optimizer, checkpoints or
convergence. They do not prove a complete transformer trainer. Timings are retained
but excluded from performance acceptance because desktop activity was not
isolated. Reproduce using `run_probes.py`, which disables optional matrix libraries
and uses CPU model loading before explicitly creating the CUDA training device:

```powershell
python backend/benchmarks/training-embedding-20260929/run_probes.py --output build-train-profile/embedding-rerun
python backend/benchmarks/training-embedding-20260929/sanitize.py --output build-train-profile/embedding-rerun
```

Use a fresh output directory for each rerun; the runners refuse to overwrite
existing logs. The inference CUDA bundle remains byte-identical: 32,423 bytes, SHA-256
`927af8f2929259b9984b8b82ddf4a327f8194a4d9603947d4758765613822d55`.

## Configuration and package validation

All [15 configuration stages](training-embedding-20260929/matrix/validation.json)
passed. CPU C17 and ASan/UBSan each passed 51 applicable tests; CUDA/Vulkan C17
and C23 each passed 97; ONNX passed 52. Physical-removal configurations passed
51 without optcpp, 51 without CUDA and 92 without CUTLASS. Each configuration
skipped four unavailable ROCm/MLX hardware checks. All 11 frontend tests and the
production frontend build passed.

The complete new embedding fixture, including matrix accumulation-order cases,
passed memcheck, initcheck, synccheck and racecheck with zero reported errors or
hazards. Every process exited zero and no launch window was excluded.
[Sanitizer commands and exits](training-embedding-20260929/sanitizers.json).

The rebuilt desktop installer is 406,274,237 bytes, SHA-256
`165b4e49c06d7ee6e6c36b9fbab4d789e907b5b82851fb538dfddb6060529c9f`.
[Package verification](training-embedding-20260929/desktop-package.json) matches
extracted backend resources to the built binaries, checks the desktop bundle
marker, passes the extracted trainer CLI lifecycle and eight loopback HTTP
requests. Native CUDA inference with optional matrix libraries disabled produces
the same eight-token continuation twice from a 725-token prompt without CPU
replay. No installation or interactive UI test is claimed.

[The final audit](training-embedding-20260929/audit.json) verifies source hashes,
all 15 matrix stages, four complete GPU sanitizer runs, five component probes,
the preserved numerical failure and fix, inference bundle identity and package
evidence. Complete GPU training and throughput improvement remain unproven.
