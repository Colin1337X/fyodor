# Weight reuse across narrow batches, 2026-11-13

Original Fyodor kernel in optcpp/matvec/small_batch_reuse.cu. Each warp reuses
one Q4_K/Q6_K decoded weight across independent input-vector accumulators,
preserving each output's column order and shuffle reduction. This differs from
the rejected adjacent-output-row experiment, which reused activations.

The isolated oracle passes all 84 cases: both formats; batches 1/2/3/4/5/8;
row tails 1/7/9/65 and model-sized gate/down/vocabulary matrices. Tile variants
1/2/4 match native row-dot scheduling byte-for-byte, with sentinel checks. Small
matrices also pass independent float64 decoding/summation. oracle.json records
all cases and the compiled source hash. No tolerance was widened.

## Isolated diagnostic timings

CUDA-event timings use repeated captured graphs, warmups, five repetitions in
each forward/reverse order, identical weights/inputs, and no concurrent evaluation
or compilation. Current desktop activity and repeated KV/weight working sets
make these diagnostics, not accepted model or llama.cpp speed claims.

| Format/shape | Batch | Native ms | Reuse ms | Speed ratio |
| --- | ---: | ---: | ---: | ---: |
| 12 / 15360 x 3840 | 2 | 0.224991 | 0.125056 | 1.799x |
| 12 / 15360 x 3840 | 4 | 0.450006 | 0.203123 | 2.215x |
| 14 / 3840 x 15360 | 2 | 0.286049 | 0.179387 | 1.595x |
| 14 / 3840 x 15360 | 4 | 0.558252 | 0.259633 | 2.150x |

## Optional runtime integration

Compile NYA_ENABLE_SMALL_BATCH_REUSE=ON with NYA_ENABLE_OPTCPP; select
NYA_CUDA_SMALL_BATCH_REUSE=1 before context creation. Both default off. Only
Q4_K/Q6_K batches exactly 2/4 select the matching tile; other formats/counts,
reference mode and absent source/symbols retain prior dispatch. Both resident
and assisted matrix interfaces share the C selector. The profile identifies
small-batch-weight-reuse separately. No extra device allocation or weight copy.

The dedicated quant-kernels-cuda-small-reuse test checks scalar bitwise equality
for supported types/counts while retaining independent float64 checks for all
formats and fallback batch sizes. It does not require the older optional
small_batch.cu file to exist. C17: sixteen selected tests pass, covering compute
tails, CPU/Vulkan/CUDA verification, ring/shared KV, failure handling and sampled
MTP comparison. C23 also passes all sixteen selected tests. CUDA memcheck reports zero errors.
The physical-removal configuration passes 82 tests (four expected unavailable
ROCm/MLX skips), train/export/infer, and a separate CUDA build plus quant tests
with optional sources absent and runtime flags enabled. Its first execution
returned skip 77 because the copied build had no NVRTC library path; the verified
rerun sets NYA_CUDA_NVRTC to the installed compiler DLL.

The initial removal build caught Windows macro leakage from the earlier profiler
clock header. That separate bug is fixed through the plain C runtime clock API;
both failed and successful removal logs remain. The compiled optional dispatch
was not responsible for that CPU-only build failure.

The same-binary complete generation off/on experiment passed in runtime-off/
and runtime-on/. Borrowed wide-value attention is disabled in both; only
NYA_CUDA_SMALL_BATCH_REUSE changes. The probe uses the recorded B512 batched
prompt ingestion and QPC decode contract. Build provenance is in build/.
All 48 calls passed exact generated text, token-count and stop-reason parity.
The first round is excluded; the remaining two observations per cell are
reported with sample standard deviations in runtime-summary.json.

| Context | Ordinary speed ratio | Serial MTP ratio | Batch-2 MTP ratio | Batch-4 MTP ratio |
| --- | ---: | ---: | ---: | ---: |
| 25 prompt tokens | 0.9991x | 1.0033x | 1.2075x | 1.3824x |
| 8205 prompt tokens | 1.0000x | 1.0038x | 1.1187x | 1.2457x |

Ratios compare reuse enabled against disabled in the same binary. Batched MTP
still takes longer than ordinary generation. The unchanged environment gate
failed, so these are diagnostic improvements, not accepted llama.cpp scores.
After timing, the kernel's opening comment was corrected to describe its new
opt-in dispatch; executable code is unchanged from the recorded source hash.
