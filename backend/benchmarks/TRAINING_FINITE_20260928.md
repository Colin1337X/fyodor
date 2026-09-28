# Resident numerical-error checks — September 28, 2026

The CPU training graph checks every forward result and backward gradient for
non-finite values. Copying complete GPU tensors to implement those checks would
defeat resident execution. The private device now has an asynchronous F32 finite
check that publishes the first failing operation's nonzero tag to a resident
32-bit status buffer. A caller reads that word at a step boundary.

## Contract and implementation

`nya_train_device_check_finite` validates handles, buffer sizes, nonzero count
and tag, non-aliasing status/data, and the grid bound before launching. It neither
allocates nor transfers data nor synchronizes. The return value reports dispatch
success; it does **not** establish numerical validity.

The caller zeroes status at the start of a validation group. A CUDA block votes
on F32 exponent bits, then one thread atomically records the tag if status is
still zero. Infinity and NaN are rejected; signed zeros, subnormals and finite
extremes are accepted. The input stays untouched. Stream ordering preserves the
first failing check's tag across later checks. Status allocated before a scratch
scope survives temporary-buffer reuse, including an overflowed matrix output
that is overwritten by a later allocation before the status is read.

Numerical failure leaves the device usable. The future graph must reject the
microbatch, discard invalid accumulated gradients and reset validation state
before retrying. Launch/driver failure still poisons the device, and even a status
read then fails. The primitive does not implement optimizer commit, numerical
recovery or automatic checking of every graph operation. CPU training remains
the CLI/UI path until the complete resident graph and transactional optimizer
are integrated and their trajectories validated.

This is original Fyodor C/CUDA code. Source study used PyTorch v2.8.0 at
`ba56102387ef21a3b04b357e5b183d48f0afefc7`, specifically
[AmpKernels.cu](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/aten/src/ATen/native/cuda/AmpKernels.cu)
and its [license](https://github.com/pytorch/pytorch/blob/ba56102387ef21a3b04b357e5b183d48f0afefc7/LICENSE).
Its AMP implementation retains a caller-reset non-finite flag on the device
across tensors. Fyodor adopts that residency principle with its own block vote
and atomic operation tag. No upstream source, ATen representation, tensor
iterator, unscale operation or runtime dependency is copied. Exact source/license
hashes are in [the snapshot](training-finite-20260928/snapshot.json).

## Correctness and resource evidence

[Five targeted CTests](training-finite-20260928/initial-tests.log) passed. The new
fixture checks eight lengths at warp/block boundaries, six representative
infinity/NaN encodings at first/middle/last positions, finite extremes, signed
zeros, subnormals, ignored out-of-range sentinels, bit-identical input preservation,
status guard bytes, first-tag preservation, all-block failure, malformed/stale/
foreign descriptors, explicit reset, and matrix overflow followed by physical
storage reuse. One hundred queued scopes add no transfers or fences. An injected
finite-check launch failure verifies that the status cannot be read as success
after device poisoning.

The full new `--finite` fixture passes memcheck, initcheck, synccheck and racecheck
without reported errors or hazards. Each process exits zero; no launch window is
excluded from these new-kernel sanitizer runs. [Commands and exits](training-finite-20260928/sanitizers.json).
The existing full matrix/storage/scratch fixture remains in normal CTest coverage.

The real TinyLlama Q4_K_M harness checks first-layer query, FFN down and FFN up
at 8 and 64 tokens, with a double oracle over 64 spread elements per result.
Checked modes queue a finite scan after every forward/dX/dW dispatch, including
two warmup and five measured batches of twenty calls, while retaining one status
buffer across six retired job scopes. That buffer is read once after all jobs.
The harness asserts that each timed batch adds no host transfers and only its
existing completion fence.

| Mode | Kernel launches | Upload bytes | Download bytes | Peak arena bytes | Final live bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reuse, unchecked | 2,826 | 42,221,568 | 4,608 | 59,531,264 | 0 |
| Checked, forward | 5,365 | 42,221,568 | 4,612 | 59,531,520 | 4 |
| Checked, reverse | 5,365 | 42,221,568 | 4,612 | 59,531,520 | 4 |

Both checked orders pass and report zero numerical-error status. The difference
is 2,538 scan kernels, one allocation-zero kernel, one four-byte download and
256 bytes of peak arena usage including alignment. All modes reserve the same
64 MiB arena. Existing oracle transfers are included; this is not a zero-transfer
end-to-end training claim. Timings are preserved but excluded from performance
acceptance because desktop activity was not isolated. These counters establish
residency cost, not throughput or convergence improvement.

Reproduce with native vendor paths disabled (`NYA_CUDA_BLAS=0`,
`NYA_CUDA_CUTLASS=0`) and CPU model loading (`NYA_COMPUTE=cpu`); the harness itself
explicitly creates a CUDA training device:

```powershell
build-cuda/nya-bench-training-device.exe .tools/tinyllama-q4_k_m.gguf reuse-forward
build-cuda/nya-bench-training-device.exe .tools/tinyllama-q4_k_m.gguf checked-forward
build-cuda/nya-bench-training-device.exe .tools/tinyllama-q4_k_m.gguf checked-reverse
python backend/benchmarks/training-finite-20260928/sanitize.py
```

The initial benchmark build caught a signed integer expression in its new launch
counter assertion under `-Werror=sign-conversion`; unsigned constants fixed it.
Both the failed build log and successful rebuild are retained. No runtime test
failure or tolerance change was involved.

The generated inference source is byte-identical to the prior module-split
snapshot (32,423 bytes, SHA-256
`927af8f2929259b9984b8b82ddf4a327f8194a4d9603947d4758765613822d55`).
Only the training program gains the new check kernel.

## Configuration validation

All 15 stages of [the validation matrix](training-finite-20260928/matrix/validation.json)
passed: CPU C17 and ASan/UBSan each passed 48 applicable tests, CUDA/Vulkan C17
and C23 each passed 81, and ONNX passed 49. Physical-removal builds passed 48
without `optcpp`, 48 without CUDA and 76 without CUTLASS. Each configuration
skipped four unavailable ROCm/MLX hardware checks. All 11 frontend tests and the
production frontend build passed.

## Desktop package

The PowerShell 7 package build completed successfully. The rebuilt
`frontend/src-tauri/target/release/bundle/nsis/Fyodor_0.3.0_x64-setup.exe` is
406,279,433 bytes, SHA-256
`fa52308fafd35e741af783aff5a031829b63fcd66cebd7515142dc089b84e602`.
[Verification](training-finite-20260928/desktop-package.json) matches extracted
backend resources against their source binaries, checks the expected Tauri
desktop bundle marker, runs the extracted trainer CLI, and exercises eight
authenticated/unauthenticated HTTP requests. Native CUDA inference with optional
matrix libraries disabled produces the same eight-token continuation twice from
a 725-token prompt without CPU replay. No installation or interactive UI test
is claimed.

[The final audit](training-finite-20260928/audit.json) verifies current source
hashes, all 15 configuration stages, four complete new-fixture sanitizer passes,
both checked real-weight orders, inference bundle identity and package evidence.
It explicitly leaves throughput acceptance and complete GPU training unproven.
