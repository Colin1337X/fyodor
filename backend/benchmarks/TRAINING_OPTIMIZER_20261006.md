# Resident transactional AdamW — October 6, 2026

The private CUDA training interface now has persistent multi-tensor AdamW plans,
global gradient clipping, an integer device step counter, and complete staging
before parameter/moment commit. A resident classifier and real TinyLlama
vocabulary-head LoRA trajectories match CPU training. This also exposed and
corrected a training matrix-forward precision mismatch that individual-operation
tests had not caught. Public CLI/UI training still uses CPU; full graph and
checkpoint ownership/integration remain unfinished.

Base revision: `234f28268e797b7160e2ae2308bdf2c4cb60db23`. Work stays on
`codex/resident-training`, preserving unrelated primary-checkout changes.
Evidence is in [training-optimizer-20261006](training-optimizer-20261006/).

## Ownership and execution

`nya_train_adamw_tensor` describes separate F32 values, gradients, first moments,
second moments and element counts. Plan creation validates all handles, sizes
and aliases; copies the caller's descriptors; uploads immutable chunk metadata;
and reserves reusable device staging in the existing bounded arena. All four
buffers for every tensor must be distinct. The sealed plan cannot be read,
written, cleared or passed as an ordinary tensor. It borrows previously allocated
buffers and follows their scratch lifetime. Destroying a device or retiring a
scope releases its host metadata; device storage retires in stream order.

For `P` tensors, `N` elements and `C = sum(ceil(elements/2048))` chunks, the plan
requires `48*P + 24*C + 32 + 12*N` arena bytes, excluding arena alignment padding.
This includes immutable descriptors/chunk mapping, double reduction scratch,
four double scalars and three complete F32 candidate arrays. The private-interface size helper rejects overflow before setup. There is no per-step
allocation, metadata upload, tensor copy or implicit fence.

Five ordered kernels perform a step, regardless of tensor count:

1. Compute double gradient-square partial sums per 2,048-element chunk and
   publish the first nonfinite-gradient tag with integer atomic compare/exchange.
2. Reduce partials in chunk order, compute the global norm, clipping factor and
   bias corrections, and reject an exhausted U64 step counter.
3. Stage all candidate values and moments. Validate finite values, F32 range and
   nonnegative updated variances before converting to F32.
4. Commit every candidate only if the entire preceding validation passed.
5. Increment the resident U64 step and publish the double gradient norm only
   after a valid commit.

A nonzero caller-owned status, including a prior loss/gradient finite check,
skips all persistent mutations. Numerical failure preserves every weight,
moment, variance, step and published norm. Gradients are never modified by AdamW.
The first status tag survives later queued attempts. Corrected inputs can be
retried on the same device after explicit status reset. Stage blocks use an
atomic status read and a block-uniform branch because another block may publish
failure concurrently.

Device or launch failure is different: it poisons the context and subsequent
reads/finish fail. A driver failure during commit cannot promise hardware-level
rollback, so recovery requires a checkpoint in a new context. The API documents
this distinction. A queued return value alone never establishes step success;
the caller observes status and step at an explicit boundary before reporting,
checkpointing or exporting.

Arithmetic follows CPU AdamW's double formulas, float configuration values,
decoupled decay, clipping and bias correction. Explicit double products/adds
prevent contraction from changing candidate-range checks. The gradient norm
uses a deterministic parallel reduction rather than the CPU's serial sum;
tests compare it at `1e-12*(1+norm)`. No mixed-precision, loss-scaling or AMSGrad
contract is introduced.

## Precision failure and correction

The first real LoRA trajectory failed the existing composed-model `1e-5` scaled
bound at the second step: B element 56,114 was `0.0133316042` on CPU versus
`0.0133424522` on GPU, scaled error `1.07053262e-5`. The failed run is preserved
in `adamw8.log` and `probes.json`; it is not accepted validation.

Resident forward matrices had reused inference GEMM's two F32 accumulation
chains. CPU dense training uses ordered double sums; its frozen low-precision
mapped matrices use ordered F32 products/additions. AdamW amplified small
forward discrepancies around nearly cancelling gradients. New training-only
32×32 tiled forward kernels now retain the CPU accumulation rules, including
Q4_0's interleaved nibble order. Inference retains its independent kernels and
unchanged embedded module. This is a correctness correction, not an inference
optimization or a claimed training throughput improvement. Dense F32 training
now incurs the cost of double accumulation; future performance work must retain
the validated training contract.

Direct matrix tests now check bit-identical ordered forward results for all
seven storage formats, tile tails and long reductions, alongside their existing
independent double error bounds. A new extreme F32 case proves that finite CPU
results remain finite when intermediate F32 products or separate accumulation
chains would overflow. No established tolerance was widened.

## Source study and provenance

Studied PyTorch revision `ba56102387ef21a3b04b357e5b183d48f0afefc7`, after reviewing
its permissive BSD-style license:

- `aten/src/ATen/native/cuda/fused_adamw_impl.cu`: parameter/gradient/moment lists
  and optional scale/overflow inputs.
- `aten/src/ATen/native/cuda/fused_adam_utils.cuh`: decoupled decay, bias
  correction and the device `found_inf` guard.
- `aten/src/ATen/native/cuda/MultiTensorApply.cuh`: mapping blocks to bounded
  chunks across many tensors.

These are original Fyodor C interfaces/host code and quarantined CUDA kernels,
informed by chunk scheduling and device guard techniques. No upstream source,
ATen tensor machinery, framework dispatch, C++ core API or new dependency was
copied. Fyodor's separate staging/commit and double numerical rules preserve its
existing CPU semantics. Exact upstream/source hashes are in `snapshot.json`.

## Tests and real-weight trajectories

`test_training_optimizer.c` compares five 20-step CPU/GPU trajectories over nine
tensor sizes: 1, 3, 255, 256, 257, 2047, 2048, 2049 and 8193. Every value, gradient,
first moment and second moment is checked against the CPU's serialized
checkpoint at the unchanged `1e-6*(1+abs(reference))` bound. Gradient bytes must
remain identical, and allocation tails retain sentinels. Cases cover no clipping,
active clipping, an inactive large clipping limit, zero betas, betas immediately
below one, decaying learning rates, decay disabled, and U64 counters beyond
double's exact integer range.

Ten transaction cases place failures in a later tensor/chunk: NaN/infinite
gradients, nonfinite values/moments, negative updated variance, overflow,
exhausted step and preexisting status. Two queued attempts must preserve all
input bytes, all persistent optimizer state and the first tag. Separate tests
reject an overflowing double candidate that would round back to `FLT_MAX`,
accept the last representable step, reject the following step, and recover
after corrected input. Descriptor tests cover aliases, undersized/stale/foreign
buffers, sealed token maps/plans and invalid settings. Arena exhaustion preserves
an existing usable plan; 100 plan/scope reuses exercise copied descriptor and
host-metadata lifetimes. Five injected failures cover every optimizer launch,
including failure after the commit kernel but before step publication.

A 50-step resident classifier composes matrix forward, CE, backward, weight
gradients and AdamW. All scalar losses and all four CPU checkpoint arrays are
compared every step; loss decreases by at least 80%. Its inner step asserts no
allocations, transfers or fences. A separate snapshot/restoration into a fresh
GPU context continues from step 10 to 20 with bit-identical values, gradients,
moments, variance, counter and norm versus uninterrupted GPU execution. This is
private optimizer-state restoration, not completed public GPU checkpointing.

`optimizer_head` in `training_device_bench.c` trains rank-four A/B adapters over
TinyLlama's real frozen Q6_K vocabulary matrix `[32000,2048]`, using fixed
synthetic hidden inputs and masked labels. CPU autograd and AdamW use six matrix
workers; GPU matrices, loss, gradients and optimizer remain resident. Each of
32 steps checks every trainable value and gradient plus the scalar loss against
CPU at the unchanged composed-model `1e-5` scaled tolerance. The tests are not
full-transformer fine-tuning or generalization evidence.

| Workload | Loss, first → last step | Maximum scaled parameter error | Maximum scaled gradient error | Peak arena bytes |
|---|---|---:|---:|---:|
| 8 tokens | 10.3229351 → 0.00000101163948 | 0 | 1.162003e-10 | 63,787,268 |
| 64 tokens | 10.3878813 → 0.288134694 | 0 | 1.058791e-22 | 107,257,348 |
| 8 tokens, reference compilation | 10.3229351 → 0.00000101163948 | 0 | 1.162003e-10 | 63,787,268 |

Maximum scaled loss error is `4.547455e-13` for the 8-token runs and zero for
64 tokens. Each optimizer plan uses 1,636,040 bytes. Setup uses seven uploads;
each step uses 21 kernels, no data copies or further arena allocation, and one
explicit finish. Seven validation downloads afterward total 224 per trajectory.
All losses and timing samples are retained in `ordered-forward-probes/`.
Timing is **not accepted as performance evidence** because the interactive
desktop was not isolated. These runs establish component trajectory parity and
residency, not full training throughput or convergence quality on a dataset.

The earlier real embedding → RMSNorm → FFN, 64-token attention branch, and
64-token DPO probes were rerun after the matrix correction. Every checked scalar,
output and gradient matched the CPU F32 values (maximum scaled errors zero).
Their complete records are in `regression-probes/`; their timings are likewise
excluded from performance acceptance.

## Validation and reproduction

All 15 stages of `matrix/validation.json` passed:

| Configuration | Applicable tests passed | Unavailable hardware skipped |
|---|---:|---:|
| GCC C17 CPU | 55 | 4 |
| GCC C17 CUDA/Vulkan | 128 | 4 |
| GCC C23 CUDA/Vulkan | 128 | 4 |
| GCC C23 ONNX | 56 | 4 |
| Clang ASan/UBSan CPU | 55 | 4 |
| Physically absent optcpp | 55 | 4 |
| Physically absent CUDA | 55 | 4 |
| Physically absent CUTLASS | 123 | 4 |

The four skips are unavailable ROCm/MLX hardware. CUDA optimizer and matrix
tests actually ran. Eleven frontend tests and the production frontend build
also passed. GCC 16.2.0 release builds retain `-O3`; the test machine is an RTX
5060 Ti 16 GB with driver 610.88 and Ryzen 5 9600X. `snapshot.json` records exact
source/model/compiler/module identities, build cache settings and engine flags.

Both complete GPU suites were checked with memcheck, initcheck, synccheck and
racecheck: eight final sanitizer runs, all zero errors, and zero race hazards or
warnings. `optimizer-final-sanitizers/` and `matrix-sanitizers/` bind those results
to the exact final test executables. Earlier optimizer sanitizer runs are
retained separately; the final suite additionally covers small first-step bias
corrections with betas immediately below one. No kernel filters or skipped
launches were used. Racecheck bounds instrumentation resources with
`--force-synchronization-limit 4 --racecheck-num-workers 2`; production execution
does not add those synchronization points.

The NSIS installer was built and extracted successfully. Every packaged backend
resource matched its build input; the desktop executable differed only by the
expected Tauri bundle marker. The packaged trainer passed the complete CLI
lifecycle test, and eight loopback HTTP checks passed, including authentication,
model loading, explicit native CUDA selection, repeated deterministic generation
with a 725-token prompt and eight generated tokens, runtime reporting and unload.
Optional cuBLAS and CUTLASS were disabled for these checks; no CPU replay or
resident-plan fallback occurred. `desktop-package.json` records installer and
resource hashes plus the responses. The installer was not installed, and the
interactive desktop UI was not exercised.

Reproduction from the configured worktree, with bundled w64devkit on PATH:

```powershell
cmake -S backend -B build-cuda
cmake --build build-cuda --parallel 6
ctest --test-dir build-cuda -R '^training-(optimizer|device)' --output-on-failure
python backend/benchmarks/training-optimizer-20261006/run_probes.py --output <fresh-trajectory-directory>
python backend/benchmarks/training-optimizer-20261006/run_probes.py --regressions --output <fresh-regression-directory>
python backend/benchmarks/training-optimizer-20261006/sanitize.py --output <fresh-optimizer-sanitizer-directory>
python backend/benchmarks/training-optimizer-20261006/sanitize.py --suite device --output <fresh-matrix-sanitizer-directory>
python backend/benchmarks/training-20260923/validate.py --output <fresh-matrix-directory>
pwsh -NoProfile -File backend/benchmarks/training-optimizer-20261006/package.ps1
python backend/benchmarks/training-optimizer-20261006/verify_package.py
python backend/benchmarks/training-optimizer-20261006/snapshot.py
python backend/benchmarks/training-optimizer-20261006/audit.py
```

Evidence logs are not overwritten; package build attempts append numbers.
Package verification requires a fresh extraction directory. The inference CUDA
module remains byte-identical to the preceding worktree milestone, SHA-256
`5d307a7bafd596eeb2f8fb886383db315edbc679562664e89bf627c56594ea4d`.
