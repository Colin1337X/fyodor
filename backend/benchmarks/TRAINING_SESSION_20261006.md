# Resident parameter ownership and portable checkpoints — 2026-10-06

This milestone connects the private resident optimizer to Fyodor's existing
parameter objects and portable checkpoints. `core/training_session.c` owns a
device and an exclusive parameter lease. It prevents a CPU graph, checkpoint or
GGUF exporter from silently consuming stale CPU copies after GPU updates.
It is **not yet a GPU implementation of the public eager graph, decoder or CLI**.
Public training and the Training UI continue to execute on CPU.

## Ownership and failure semantics

Creation validates settings, every value/gradient/moment/variance, unique
parameters, and absence of existing CPU leaves or resident leases. It copies all
four arrays to persistent device buffers, creates the sealed AdamW plan and
uploads the completed-step counter. Only fully successful initialization takes
the lease; failed allocation, upload or launch leaves CPU parameters untouched
and usable. Session creation does not retain caller descriptor arrays.

While leased, CPU data and gradient access return NULL, CPU zero-grad does
nothing, and CPU leaf creation, AdamW, checkpoint read/write and decoder export
reject the parameters. Export checks every parameter before writing any GGUF
bytes, including imported-model paths. Previously borrowed host pointers must
not be used during the lease. This is a serialized C API, not a thread-safe
mutation protocol. GPU graph code borrows session handles and must obey their
lifetime rules; it cannot free the device or overwrite optimizer metadata.

Parameter storage is reference counted internally. CPU leaves and sessions keep
it alive until their own destruction, even if the caller has released its
parameter reference. CPU leaves are counted separately so a new resident session
cannot invalidate a graph's borrowed data or gradients. CPU graphs still require
values to remain unchanged through backward; retaining storage is not permission
to concurrently mutate it.

A session queues gradient/status reset and one AdamW update without host copies,
allocation or a fence. The caller must observe the pending result before another
update or reset. Observation explicitly reads status, integer step and gradient
norm. Only status zero and the expected counter publish the new optimizer
settings in the session's host metadata. A numeric failure retains the prior
settings and step; correction plus explicit gradient/status reset permits retry.
Launch/device failures poison the device and require discarding that session.

Detach first stages **all four arrays for all parameters**, validates finite
values and nonnegative variance, then publishes the entire CPU state and optimizer
settings/counter. A failure in a late tensor cannot refresh earlier CPU tensors.
Successful detach releases the device and lease; the session object can then be
freed. Freeing without successful detach discards all GPU updates and returns the
original, unmodified CPU recovery snapshot. It never attempts an implicit best
effort copy after a device failure.

## Checkpoint compatibility and memory

`nya_train_session_checkpoint_write` performs the same staged readback and calls
the existing CPU serializer on independent temporary parameter descriptors. It
never temporarily unlocks or changes live CPU parameters. Numeric/state failure
is detected before writing the checkpoint prefix. File I/O failures can still
leave a partial output stream; atomic file replacement remains the caller's
responsibility, as with the existing CPU API.

The checkpoint format is unchanged: values, accumulated gradients, first and
second moments, optimizer settings and the integer step use the original
versioned little-endian representation and checksum. Resume loads that checkpoint
with `nya_train_checkpoint_read` before creating a fresh session. The diagnostic
last gradient norm starts at zero in a new session; it is not serialized in this
format. This bridge does not implement runner dataset/RNG/scheduler state; the
existing CLI's separate run identity remains unchanged.

For N parameter elements, existing host recovery arrays retain 16N bytes;
resident arrays use another 16N bytes, and transactional AdamW reserves 12N bytes
plus plan metadata and alignment. Checkpoint/detach temporarily need an additional
16N host bytes plus parameter descriptors. Graph activations, mapped frozen
weights, context/module overhead and driver allocations are separate. The caller
sets the device arena budget; this is not a hard process-RAM ceiling. Snapshot
readback uses four array transfers per parameter plus the three scalar observation
reads, and occurs only at an explicit checkpoint/detach boundary. A normal
optimizer observation reads only the three scalars. Transfer consolidation and
releasing redundant host recovery storage remain possible later optimizations.

## Validation contract

`test_training_session.c` covers:

- Null, malformed and duplicate descriptors, invalid settings/host values,
  unsupported devices and failed arena admission, with no leaked leases.
- Live training/evaluation leaves, simultaneous-session rejection, CPU access
  guards and rejection before checkpoint stream consumption or output.
- CPU graph retention after the caller frees a parameter; session retention
  after caller release; repeated session/scratch/detach lifetimes.
- Numeric gradient rejection, sticky first failure tags, explicit recovery,
  exhausted U64 step counters and invalid values in each snapshot array.
- A late invalid variance after an earlier GPU tensor changed, proving failed
  detach plus abort leaves the complete original CPU checkpoint byte-identical.
- Two constructor launch failures and all five optimizer launch failure positions,
  including failure after persistent GPU values changed but before counter
  publication. No failed session publishes partial host state.
- A 40-step classifier using dense forward, broadcast bias, CE, backward,
  two accumulated microbatches, clipped AdamW and changing learning rates.
  Every loss and all four parameter arrays match CPU at the unchanged `1e-6`
  scaled tolerance after every update. Dispatch checks assert no inner host
  transfers, new arena allocation or synchronization.
- A checkpoint at step 10, destruction/reinitialization of GPU state, and 30 more
  updates yielding a **byte-identical** final checkpoint to uninterrupted GPU
  execution. A successful detach produces those exact checkpoint bytes too.
- A random LLaMA decoder refuses GGUF export while leased, without writing a
  prefix, and exports successfully after detach.

These are state-management and component trajectory checks. They do not prove
full-transformer GPU training, useful dataset generalization, or training
throughput. Timing from this interactive desktop is not used for performance
acceptance. Existing inference source and CUDA training kernels are unchanged.
The bridge is original Fyodor C code using existing interfaces and checkpoint
semantics; it introduces no foreign runtime or copied upstream source.

The first build was rejected by strict warnings for an implicit unsigned-to-float
test conversion and misleading indentation. Both test-source defects were fixed;
the failed build log is preserved, and no tolerance or warning policy changed.

## Reproduction

From the configured worktree with bundled w64devkit on PATH:

```powershell
cmake -S backend -B build-cuda
cmake --build build-cuda --parallel 6
ctest --test-dir build-cuda -R training-session --output-on-failure
python backend/benchmarks/training-session-20261006/sanitize.py --output <fresh-sanitizer-directory>
python backend/benchmarks/training-20260923/validate.py --output <fresh-matrix-directory>
pwsh -NoProfile -File backend/benchmarks/training-session-20261006/package.ps1
python backend/benchmarks/training-session-20261006/verify_package.py
```

## Completed validation

Validation completed on 2026-10-07. All 15 stages passed. Every configuration
skipped only the four unavailable ROCm/MLX hardware tests; CUDA session execution
and injected failures actually ran.

| Configuration | Applicable tests passed | Unavailable hardware skipped |
|---|---:|---:|
| GCC C17 CPU | 56 | 4 |
| GCC C17 CUDA/Vulkan | 138 | 4 |
| GCC C23 CUDA/Vulkan | 138 | 4 |
| GCC C23 ONNX | 57 | 4 |
| Clang ASan/UBSan CPU | 56 | 4 |
| Physically absent optcpp | 56 | 4 |
| Physically absent CUDA | 56 | 4 |
| Physically absent CUTLASS | 133 | 4 |

Eleven frontend tests and the production frontend build passed. GPU memcheck,
initcheck, synccheck and racecheck each ran the complete session suite and
reported zero errors; racecheck reported zero hazards and warnings. Instrumented
racecheck bounds resource use with `--force-synchronization-limit 4
--racecheck-num-workers 2`, without skipping launches or filtering kernels.
Normal execution asserts no corresponding inner fences.

The accumulated two-microbatch classifier loss went from 2.96077681 to
1.96025687. This is the sum of two mean losses. Every checked loss and all
parameter/gradient/moment/variance values met `1e-6` CPU scaled tolerance; the
step-10 checkpoint resumed to a byte-identical step-40 checkpoint. Sanitizer
records bind these checks to executable SHA-256
`5a51f235fc6cb68a36ed0b35716a644b86c9985fc60e2060c2e110a74bc9784c`.

`snapshot.json` records source hashes, GCC 16.2.0 and release `-O3` configuration,
RTX 5060 Ti 16 GB/driver 610.88, executable/model hashes and both embedded CUDA
module hashes. Both the inference and training CUDA modules remain byte-identical
to the preceding optimizer milestone. This work adds no inference performance
claim. Evidence is under `training-session-20261006/`; earlier logs are retained.


The NSIS package built and extracted successfully. Packaged backend resources
matched their build inputs byte for byte; the desktop executable matched after
the expected Tauri bundle marker substitution. The packaged native trainer
passed its full CLI lifecycle checks. Eight authenticated/unauthenticated
loopback HTTP checks passed, including native CUDA generation twice with a
725-token prompt and eight generated tokens, with identical output, runtime
reporting and model unload. Optional cuBLAS/CUTLASS were disabled for that
inference check and no CPU replay or resident-plan fallback was reported.
`desktop-package.json` records resource and installer hashes and responses.
The installer was not installed and interactive desktop UI behavior was not
exercised. `audit.py` verifies the completed evidence against the current
source and binary hashes; run it after `snapshot.py` and package verification.
