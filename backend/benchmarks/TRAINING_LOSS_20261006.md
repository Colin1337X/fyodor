# Resident masked losses and DPO — October 6, 2026

The private CUDA training interface now evaluates masked hard-label
cross-entropy, summed selected log probabilities, and scalar DPO, with their
gradients remaining on the device. Real TinyLlama vocabulary projections compose
with both losses and propagate two microbatches into input gradients matching
CPU autograd. Public CLI/UI training still uses CPU: transactional GPU AdamW,
graph ownership and full training integration remain unfinished.

Base revision: `068af7c877201e5c74d97c22042838ebe280a673` on
`codex/resident-training`. Evidence and replay scripts are in
[training-loss-20261006](training-loss-20261006/). The primary checkout's unrelated
work was not changed.

## Implementation and numerical contract

`training_device.h/.c` and its private C backend interface add loss/DPO dispatch.
Host validation stays in `CUDA/training.inc`; device kernels stay in
`CUDA/training.cu`. There is no framework dependency, C++ ABI exposed to the core,
or implicit allocation/copy/fence during loss dispatch.

For logits `[rows,vocabulary]`, labels are resident U32 IDs, with an optional
byte mask. Any nonzero mask value selects a row. CE averages only selected rows;
logprob sums them, as required by Fyodor's DPO semantics. Masked invalid labels
are allowed. Out-of-range active labels never index the logits; they publish a
nonfinite result for the explicit finite-status check. An empty active set also
produces a nonfinite scalar. Neither case poisons the device, so corrected inputs
can be evaluated again. Graph integration must reject a failed step before any
optimizer mutation and still validate its upstream tensors.

One block per row computes a double maximum and double exponential sum. Shared
reduction storage is fixed at 256 doubles. The saved state has four doubles per
row (maximum, mass, selected log probability, active flag), plus one double loss
coefficient: `32*rows+8` bytes, independent of vocabulary size. A single-thread
second kernel sums selected row contributions in CPU row order. Backward uses
the saved normalization and a resident scalar seed, rounding each contribution
to F32 before accumulating into existing gradients. It recomputes probabilities
instead of storing a full vocabulary-sized matrix. Original inputs, labels,
mask and state must remain unchanged until backward finishes.

DPO uses a stable double softplus and saves one double derivative. Reference
log probabilities remain host-supplied doubles; beta is positive finite F32.
Nonfinite margins (including overflowing subtraction of finite references) and
unrepresentable F32 losses are caught by scalar finite checks. Backward accepts
either gradient or both. Shared chosen/rejected gradient storage preserves the
CPU's rounded add-then-subtract sequence, rather than cancelling contributions
algebraically. Gradient buffers cannot alias their consumed inputs/state.

Descriptors reject zero dimensions, overflow, insufficient storage, forbidden
aliases, stale/foreign handles and sealed embedding maps before launching work.
Launch failures poison the context, and subsequent reads/finish fail instead of
returning plausible partial results.

## Source study and provenance

Studied PyTorch revision `ba56102387ef21a3b04b357e5b183d48f0afefc7`:

- `aten/src/ATen/native/cuda/SoftMax.cu`: per-sample block reductions and stable
  max/sum log-softmax epilogues, including `cunn_SoftMaxForward`.
- `aten/src/ATen/native/LossNLL.cpp`: ignored-label handling, total-weight
  normalization and the hard-label cross-entropy path through log-softmax/NLL.

PyTorch's permissive BSD-style `LICENSE` was read before implementation. File
hashes and the pinned revision are in `snapshot.json`. These are original Fyodor
kernels informed by those reduction techniques; no upstream code was copied.
Fyodor keeps its own double arithmetic, byte-mask semantics and accumulation
order, without importing ATen dispatch or tensor machinery. DPO follows Fyodor's
existing CPU implementation.

## Correctness and resource evidence

`test_training_loss.c` covers 96 ordinary CE/logprob cases: rows 1/3/65,
vocabularies 1/2/7/255/256/257/1025/32000, masked/unmasked, and both reductions.
Four further cases exercise equal `FLT_MAX` logits and widely separated logits.
Every scalar and gradient is checked against CPU autograd at the unchanged
`1e-6*(1+abs(reference))` bound, including two accumulations over nonzero initial
gradients. Independent double finite differences cover masked CE/logprob and a
complete paired logprob-to-DPO calculation. Twenty scalar DPO cases cover both
partial gradients, both distinct gradients, shared gradients, saturated margins,
extreme finite references and subnormal beta. Shared accumulation is also checked
bit-for-bit against the CPU rounding sequence.

Further tests cover buffer-tail sentinels, invalid descriptors, numeric status,
masked nonfinite rows, empty masks, active invalid labels, infinities/NaNs,
loss overflow, and same-context recovery. One hundred scratch-scope reuses issue
700 kernels with unchanged persistent storage and no host transfers/fences.
Five injected failures cover both loss-forward kernels, loss backward, DPO
forward and DPO backward. All eight focused CTest entries passed on the GPU,
including reference compilation mode.

The initial focused invocation skipped GPU tests because the NVRTC source used
undefined `NAN`/`INFINITY` macros. Debug compiler output identified that setup
error; the kernels now use the same explicit bit construction as existing CUDA
code. `initial-tests.*` is retained as failed validation evidence, not counted
as a GPU pass. The subsequent complete GPU suite and all sanitizers ran normally.

All four CUDA sanitizers passed the complete final loss suite: memcheck,
initcheck and synccheck report zero errors; racecheck reports zero hazards or
warnings. Racecheck uses `--force-synchronization-limit 4
--racecheck-num-workers 2` to bound instrumentation resources, following the
previous attention milestone. This synchronization is sanitizer-only; ordinary
tests assert dispatch does not add fences. Logs, executable SHA-256 and commands
are in `sanitizers/`.

The real-model probe uses TinyLlama Q4_K_M's actual Q6_K output matrix,
`[32000,2048]`, at 8 and 64 rows. Masked CE and paired DPO each accumulate two
microbatches. A six-worker CPU graph supplies the reference. Two warmups and
five measured repetitions check every logit, scalar loss and input gradient,
using the existing composed-model scaled tolerance `1e-5`.

| Probe | Maximum scaled logit error | Maximum scaled input-gradient error | Peak arena bytes |
|---|---:|---:|---:|
| CE, 8 rows | 2.042396e-6 | 7.317869e-9 | 55,940,864 |
| CE, 64 rows | 2.108876e-6 | 1.393115e-9 | 71,196,160 |
| DPO, 8 rows | 2.042396e-6 | 2.361341e-8 | 58,121,480 |
| DPO, 64 rows | 2.108876e-6 | 2.361341e-8 | 88,632,072 |
| DPO, 8 rows, reference compilation | 2.042396e-6 | 2.315041e-8 | 58,121,480 |

Scalar losses match CPU F32 values in all five probes. Saved loss state per
branch is 264 bytes at 8 rows or 2,056 bytes at 64 rows. Setup uses five uploads
for CE or six for DPO. The two-microbatch computation uses 20 or 56 kernels,
respectively, no copies, no additional live arena storage after scopes retire,
and one explicit finish. Validation downloads occur only afterward: 28 total
for CE, 42 for DPO across seven repetitions. `final-probes/` records the exact
final executable; initial probe evidence is retained separately.

Raw timing samples are retained but **not accepted as performance results**:
the interactive desktop was not isolated. This demonstrates numerical parity,
bounded saved state and residency, not faster end-to-end training, optimizer
correctness, convergence or a full-model GPU trajectory.

## Build, removal and package validation

All 15 stages of `matrix/validation.json` passed:

| Configuration | Applicable tests passed | Unavailable hardware skipped |
|---|---:|---:|
| GCC C17 CPU | 54 | 4 |
| GCC C17 CUDA/Vulkan | 120 | 4 |
| GCC C23 CUDA/Vulkan | 120 | 4 |
| GCC C23 ONNX | 55 | 4 |
| Clang ASan/UBSan CPU | 54 | 4 |
| Physically absent optcpp | 54 | 4 |
| Physically absent CUDA | 54 | 4 |
| Physically absent CUTLASS | 115 | 4 |

The four skips are unavailable ROCm/MLX hardware, not CUDA skips. The matrix also
includes 11 frontend tests and a production frontend build. Core release builds
use GCC 16.2.0 and `-O3`; the machine has an RTX 5060 Ti 16 GB, driver 610.88 and
Ryzen 5 9600X. Source/model/compiler/module identities are recorded in
`snapshot.json`; exact compiler/build options remain in the matrix logs/caches.

Reproduction from the configured worktree, with bundled w64devkit on PATH:

```powershell
cmake -S backend -B build-cuda
cmake --build build-cuda --parallel 6
ctest --test-dir build-cuda -R '^training-loss' --output-on-failure
python backend/benchmarks/training-loss-20261006/run_probes.py --output <fresh-probe-directory>
python backend/benchmarks/training-loss-20261006/sanitize.py --output <fresh-sanitizer-directory>
python backend/benchmarks/training-20260923/validate.py --output <fresh-matrix-directory>
pwsh -NoProfile -File backend/benchmarks/training-loss-20261006/package.ps1
python backend/benchmarks/training-loss-20261006/verify_package.py
python backend/benchmarks/training-loss-20261006/snapshot.py
```

Probe/sanitizer logs are never overwritten; package builds append attempt
numbers. Package verification requires its extraction directory to be absent.
The inference CUDA module remains byte-identical to the preceding worktree
milestone, SHA-256
`5d307a7bafd596eeb2f8fb886383db315edbc679562664e89bf627c56594ea4d`.
This change makes no inference speed claim.

The NSIS package was built, extracted and verified. Its backend/trainer and
runtime-library hashes match the supplied build artifacts; the desktop binary
differs only by Tauri's expected UNK-to-NSS bundle marker. The packaged trainer
passed the CLI training/checkpoint/resume/export lifecycle. Eight authenticated
loopback HTTP checks covered authentication rejection, health, model load,
native CUDA selection, repeatable 725-token-prompt/eight-token generation,
runtime inspection and unload. Optional BLAS/CUTLASS were disabled and no CPU
prefix replay occurred. `desktop-package.json` records hashes and responses.
The installer was not installed or interactively UI-tested. Generated packaging
changes were restored after verification; packaged binaries are not committed.
