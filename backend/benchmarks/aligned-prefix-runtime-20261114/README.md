# Optional aligned prefix integration

This integrates the isolated 2K/8K prefix experiment and batched FDPP kernel.
It is default off at build and runtime and has no accepted throughput score.
Accepted Gemma decode ratios remain 85.61025294% short and 106.8707255% long
of the historical equivalent llama.cpp baseline; the 125% goal remains open.

Build with NYA_ENABLE_ALIGNED_PREFIX=ON, NYA_ENABLE_OPTCPP=ON,
NYA_ENABLE_PORTABLE_ATTENTION=ON, NYA_ENABLE_WIDE_VALUE=ON,
NYA_ENABLE_WIDE_BATCH_SPLIT=ON, NYA_ENABLE_SMALL_BATCH=ON and
NYA_ENABLE_SMALL_BATCH_REUSE=ON. Missing required optional files omit the
feature. Runtime additionally requires NYA_CUDA_ALIGNED_PREFIX=1,
NYA_CUDA_WIDE_VALUE=1, NYA_CUDA_WIDE_BATCH=1, NYA_CUDA_WIDE_BATCH_SPLIT=1,
NYA_CUDA_WIDE_SPLIT=1, NYA_CUDA_SMALL_BATCH=1,
NYA_CUDA_SMALL_BATCH_REUSE=1 and NYA_CUDA_FLASH_DECODING_PP=short.
Do not disable split attention or select portable FDPP.

Admission checks the measured dense Gemma geometry and Q4_K/Q6_K projection
types, capacity >256, active small/reuse kernels and required attention kernels.
Full-attention layers may copy projected keys as values, as the real model does.
Other geometries, small capacities, absent components and unavailable optional
symbols retain existing execution. Missing symbols may leave a conservatively
larger scratch reservation, but never cause a call through a missing pointer.

Large prefix batches use the existing four-input weight-reuse kernel with the
serial dot reduction order, instead of vendor/native tiled GEMM. Narrow batches
retain their existing row-dot/interleaved projection dispatch. Before position
256, batched 256-wide attention uses the serial FDPP arithmetic. Later prefix
batches use per-query wide split attention. Both prefill and verification stop
chunks at position 256. Serial decode and selective long verification attention
are unchanged. Query-major partial storage is checked before multiplication and
reduced with the admitted batch size under memory pressure.

## Evidence

run_verify.py retains test_verify.c lifecycle assertions and the 0.0005 bound,
specializing only the token count. It uses the actual integrated library and
records source/library/binary hashes, settings, exit status and observed admission.
Original files cannot be overwritten; an explicit run label creates new evidence.
Interleaved two/four verification and batched output projections are enabled.

- Initial prefix255/ring1/count4: numerical assertions pass, but alignment was
  not admitted. The runner correctly rejects this as feature validation. The
  guard incorrectly required a separate value matrix on full-attention layers.
  This rejected artifact is retained, not presented as an aligned-path pass.
- Corrected prefix255/ring1/count4-v2: admitted, both resets pass, maximum scaled
  logit error 0.0000178964637. Exercises a verification batch crossing position
  256 and a three-token tail. FDPP batch calls are observed.
- Prefix31/ring0/count4: intentionally not admitted (capacity36), both resets
  pass, maximum scaled error 0.0000524084294. Small-capacity fallback is observed.
- Prefix257/ring1/count4: admitted, both resets pass, maximum scaled error
  0.0000219546455. Existing interleaved verification calls remain observed.
- Integrated prefix8192/ring1/count4: admitted at batch512, both resets pass,
  maximum scaled error 0.0000163989979. Rewind and replacement-token replay
  retain the original assertions. The candidate records 80 FDPP batch calls,
  1552 batched split-attention calls and 658 interleaved verification calls.
- The isolated prior prefix8192/ring0/count2 lifecycle now passes with zero
  logit difference on both resets, including rewind and replacement replay.
  Its evidence remains in ../fdpp-batch-20261114. It is distinct from this
  integration, which preserves more of the existing verification dispatch.

Strict -O3 C17 and C23 library builds pass with warnings as errors. The
verify-without-non-c target passes all 82 tests with four expected ROCm/MLX
hardware skips, plus CPU inference/train/export lifecycle checks. Its copied
tree physically lacks optcpp, backend/optcpp, CUDA/cutlass and CUDA/cute.
A separate C23 CUDA-host build in that same removed-source tree also passes
with all aligned-prefix prerequisites requested ON; CMake reports the feature
omitted. The removal target now explicitly requests NYA_ENABLE_ALIGNED_PREFIX.
removal.log and removal-cuda-c23.log retain those build/test outputs.

admission.c/run_admission.py also check actual model plans without inference.
At capacity260, alignment reserves 260 partial-query slots. Its measured scratch
increase over feature-off is exactly 33,685,504 bytes, matching the additional
256 queries beyond the ordinary four-query reserve. Disabling weight reuse,
split attention or warp FDPP prevents aligned admission while retaining a
resident plan. With capacity8195 and NYA_CUDA_MEMORY_MIB=8500, the planner reduces
batch512 to batch256 and reserves exactly 256 partial-query slots. These checks
are asserted by summarize_admission.py; metadata and raw logs are retained.
The cache budget is not a hard process-wide VRAM limit: vendor storage is
accounted separately. The admission probe compiles under strict C17 and C23.

run_verify.py now supports --memcheck. A real prefix513/ring1/count4 check is
running through Compute Sanitizer, exercising batched FDPP and a 257-query
long split prefix chunk. The launch version is preserved verbatim as
run_verify_memcheck_v1.py.txt; after observing sanitizer output on stdout,
the next-run parser was corrected to inspect both output streams. If the live
launch reports a parser failure, its raw logs and child status must be audited
separately rather than rerunning or declaring sanitizer success from a banner.

Additional long/linear and boundary shapes, integrated Compute Sanitizer,
inference under constrained admission,
and equivalent uncontaminated generation comparisons remain required before
any performance acceptance or default enablement. Raw batched FDPP and wide
split memory checks from earlier artifacts do not substitute for these gates.

## Gated generation diagnostic

build_generation.py compiled an isolated B512 prompt-ingestion diagnostic with
strict C17 and C23 flags; generation-build.json records commands and hashes.
The production library is unchanged. The executable includes native QPC decode
intervals for joining continuous GPU activity to the measured generation loops.
No generation timing has yet been collected from this build.

run_generation.py requires the completed integrated sanitizer audit and verifies
both build and sanitizer input hashes before launching short/long diagnostics.
It enables continuous PDH monitoring through the existing MTP-cost runner.
A negative launch check while sanitizer metadata was absent rejected execution
without creating an output directory. Six offline audit tests pass, covering
missing completion, malformed error summaries, lifecycle/dispatch omissions,
and nonfinite or excessive numerical errors.

audit_memcheck.py supplements the original launch metadata only after completion;
it checks both untouched log streams, both lifecycle resets, the sanitizer's
zero-error summary, and launch input hashes. It does not rewrite the original
stderr-only parser result. The 513-token test does not wrap the sliding ring.
Generation timing remains a separate contract from fixed-forward llama-bench:
these diagnostics cannot directly replace accepted cross-engine percentages.

## External llama.cpp generation probe

llama_generation.c is an independent C reference executable, outside Fyodor's
build and runtime. It uses b10809's public API with the same short/long prompt
text, B512 ingestion, F32 KV, disabled Flash Attention, greedy decoding, and
32 output tokens including the sample from final prompt logits. It requires
CUDA0 and refuses CPU fallback. Three rounds retain the initial warmup separately.
Prompt IDs, generated IDs, output bytes (hex), EOS, and QPC endpoints are emitted
outside the measured loop. Cache reset and prompt ingestion are outside decode.
The loop includes greedy selection and token-to-piece conversion; Fyodor also
performs final UTF-8 validation, so this small accounting difference is explicit.

build_llama_generation.py compiles strict C17/C23 binaries without running them,
and records the upstream revision plus header, DLL, source, and binary hashes.
Use the recorded external DLL directory in PATH when executing. This is original
benchmark glue against the MIT-licensed public API, not an embedded llama runtime.
Prompt-token equality, greedy-output agreement, live GPU coverage, and controlled
repeated performance comparison are still unverified. No performance acceptance
or replacement of existing fixed-forward scores follows from these builds.

## Prompt-token equivalence

compare_prompt_tokens.py builds tokenize_prompt.c under strict C17 and C23 for
each implementation, then executes the C17 variants against the real Gemma model.
Fyodor links the CPU-only library; the external b10809 CPU library loads vocabulary
only and explicitly skips tensors. Neither comparison performs GPU inference.
The script checks the exact fragments in both generation prompt constructors.

The short prompt produces 25 identical token IDs and the long prompt produces
8,205 identical token IDs. prompt-token-parity/result.json retains commands,
exit codes, hashes and exact token lists. This establishes input token equality
for these two generation workloads, not generated-output or logits parity.
The long generation prompt is 8,205 tokens and must not be mislabeled as the
older fixed-forward 8,192-token context. No throughput score is changed.

run_llama_generation.py now prepares continuous PDH observation for the external
reference. It requires completed integrated sanitizer proof and both completed
native context runs, verifies recorded build/tokenizer inputs, and refuses an
existing output directory. The existing strict idle gate runs before launch.
Native child lifetime bounds, GPU/CPU/thermal snapshots, exact output, and sampled
coverage are retained even when later validation rejects the diagnostic.
No benchmark acceptance follows automatically from clear sampled coverage.

llama_intervals.py verifies the exact prompt IDs, three ordered complete rounds,
32 non-EOS outputs, finite positive QPC intervals inside the child lifetime,
duration/endpoint consistency, valid output bytes, and stable greedy output.
Six offline tests pass, including malformed/partial evidence and nondeterminism.
The live runner's negative preflight test rejects the still-pending sanitizer
without creating its output directory or starting GPU work. Actual reference
execution and cross-engine generated-output comparison remain pending.

compare_generation.py joins the two completed audits after rechecking their raw
input hashes. It requires both contexts, complete round/mode inventories, equal
prompt/output counts, and identical generated text bytes. It reports warmed
mean/sample deviation and diagnostic Fyodor speed as 100*llama_ms/Fyodor_ms for
each mode. Round zero is excluded. Coverage and legacy idle results remain
explicit, and accepted_benchmark is always false: this initial engine ordering
and two warmed observations cannot establish the requested accepted score.

Four offline comparison tests pass (warmup/ratio direction, text boundaries,
workload mismatch, invalid times). A negative CLI check rejects missing complete
audits without writing a comparison artifact. Real comparison remains pending;
no synthetic test value is a measured runtime performance result.
