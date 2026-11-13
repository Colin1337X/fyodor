# Q4 integer products in interleaved batch-four projections

This is an unintegrated original Fyodor experiment in
`optcpp/matvec/interleaved_integer.cu`. It derives from the existing interleaved
batch-four kernel and emits only a Q4_K specialization. No CMake or runtime
dispatch changes are made, and no production library is rebuilt.

The measured historical profiles identify target matrix operations as a major
decode cost. The interleaved input path improved batch-four generation in
`../interleaved-reuse-20261114`, while launch-bound variants had smaller mixed
results in `../reuse-launch-shapes-20261114`. Inspection found that interleaved
Q4 reconstruction still computes `d * (float)sc * (float)q`, unlike the previously
validated single-token integer-product path.

The candidate replaces only that expression with `d * (float)(sc * q)`.
Six-bit scale times four-bit nibble is at most 945. Products of a finite F16
coefficient and these integers fit exactly in F32. The prior exhaustive CPU
check covers 65,011,712 combinations including signed zero; see
`../decode-integer-scale-20261027/coefficient-check.json`. This justification
does not include nonfinite model coefficients and does not substitute for
testing NVRTC contraction in the new kernel.

Weight storage, minimum subtraction, input interleaving, each output's
accumulation order, and warp reduction are preserved. Q6 is deliberately not
instantiated: its signed integer product requires separate signed-zero handling.

`compile_only.py` checks the exact source transformation and compiles native and
candidate symbols together with NVRTC for compute_120, C++11, fmad enabled and
the existing fast-math define. It never loads the CUDA driver, creates a device
context, or launches a kernel. Compilation passes; hashes, flags and the compiler
log are retained. This CPU-side compilation ran while the separate integrated
aligned-prefix sanitizer job remained live, not during a throughput measurement.

Raw byte-exact comparisons, output guards, memory sanitization, full-model
parity and paired short/long generation timing are pending. The first GPU test
should extend the existing interleaved raw sweep over all eight profiled Q4
shapes and small row tails, with finite coefficient edge cases. Do not enable
the candidate from compilation or arithmetic reasoning alone. No speedup is
claimed and accepted llama.cpp scores remain unchanged.

## Prepared raw correctness sweep

check_kernel.py now prepares 207 cases: the eight profiled Q4 shapes plus three
small row-tail shapes, followed by all 196 pairs from fourteen finite half-bit
patterns at 256 columns and nine rows. Patterns include both signed zeros,
minimum/maximum subnormals, minimum normals, units, maximum finite magnitudes,
and non-binary values. Quantized metadata/nibbles and four activation vectors
are deterministic pseudorandom inputs.

Both existing interleaved Q4 and the candidate must match native small-batch
output bytes and guards. The conversion is independently checked against a CPU
transpose with prefix/suffix guards. All small cases retain the existing float64
bound of 2e-6 times absolute-product sum plus 1e-6; the bound is not relaxed.
The harness records partial evidence on test failure and refuses an existing
result. common.py is derived from the existing interleaved diagnostic helper,
adding only the candidate source and symbol to its compilation/function list.

Syntax checks passed. A negative launch preflight correctly stopped before the
GPU-creating import while the integrated sanitizer audit was absent; no oracle
file was produced. The sweep has NOT run. Once the GPU is free, execute
check_kernel.py for raw parity; a separate Compute Sanitizer invocation of
check_kernel.py --memcheck writes memcheck-oracle.json. That flag only selects
the output name: zero sanitizer errors must be established from the actual
sanitizer process status and complete log, never inferred from the filename.

## Prepared kernel timing

`probe.py` compares existing interleaved Q4 against the integer-product candidate
on all eight profiled Q4 shapes. It checks the successful 207-case oracle and
its source/harness hashes before importing the GPU helper. A missing oracle
was verified to reject launch without producing a timing artifact.

Both paths include input conversion. Each CUDA graph contains ten conversion
and projection pairs; after thirty warmup pairs, each event sample measures
one hundred pairs. Five samples per path in each forward/reverse order retain
ten observations, means and sample standard deviations. Output bytes must
agree across timed paths, and event durations must be finite and positive.
Results record compilation options, hashes and environment snapshots.

This repeated-weight synthetic measurement is diagnostic only, with
`accepted_benchmark=false`; it has no continuous interval contamination audit
and cannot establish the llama.cpp target. It has not executed. Raw correctness,
actual sanitizer validation and full-model generation remain prerequisites to
production admission. No runtime dispatch or production build changes are made.

## CPU-only PTX inspection

`inspect_ptx.py` recompiles with the recorded options without loading the CUDA
driver and requires the exact earlier PTX hash. Its instruction census finds
24 versus 16 `mul.f32` sites, with eight `mul.lo.s32` sites added. Eight
`cvt.rn.f32.u16` sites become `cvt.u32.u16`; both retain sixteen
`cvt.rn.f32.u32` sites, 32 FMA sites and twenty shuffle sites. Thus this is an
arithmetic-pipeline substitution, not an overall instruction-count reduction.

The census is static PTX, not executed instruction counts, final SASS, physical
register allocation, occupancy, numerical proof or timing. Keep the candidate
unintegrated until the raw/model checks and paired measurements establish value.
The JSON retains both opcode inventories and virtual-register declarations.

## Prepared sanitizer launcher

`run_memcheck.py` runs the 207-case sweep under the actual Compute Sanitizer
executable with `--tool memcheck --error-exitcode 99`. It waits for the integrated
sanitizer audit and ordinary raw oracle before launch, checks matching source
and harness hashes, and refuses existing evidence. It retains separate stdout
and stderr, the child exit code, elapsed time, source/tool hashes and raw result
hashes. Success requires exit zero, exactly one zero-error summary across both
streams, complete byte parity and guards, and unchanged inputs.

Five offline tests cover both summary streams, failed/invalid exits, missing,
nonzero or duplicate summaries, incomplete sweeps and failed guards. The pending
audit preflight was verified to reject launch without creating output files.
This launcher has not executed a GPU test and does not establish performance.

## Completed GPU validation and diagnostic timing

The later execution supersedes the pending status above. All 207 raw cases passed; the actual Compute Sanitizer repeat also passed all 207 with zero errors. Source and harness hashes matched.

After closing Firefox, Thorium, Free Download Manager and PowerToys, the repeated-weight timing probe completed in both orders. These are diagnostic kernel measurements, not accepted full-model decode scores; continuous interval coverage was not collected.

| Columns | Rows | Baseline / candidate speed ratio |
| --- | --- | --- |
| 3840 | 512 | 1.0091x |
| 3840 | 2048 | 1.0621x |
| 3840 | 4096 | 1.0614x |
| 3840 | 8192 | 1.0693x |
| 3840 | 15360 | 1.0620x |
| 4096 | 3840 | 1.0543x |
| 8192 | 3840 | 1.0598x |
| 15360 | 3840 | 1.0584x |


The candidate remains unintegrated pending model-level parity and measured benefit.

## Opt-in runtime experiment

NYA_ENABLE_INTERLEAVED_INTEGER compiles the optional kernel only when optcpp,
interleaved_reuse.cu and interleaved_integer.cu are physically present and batch-four
reuse is enabled. Its default is OFF. NYA_CUDA_INTERLEAVED_INTEGER=1 selects it
per resident plan; unset/zero keeps the existing kernel. Missing symbols retain
the existing path. Training and reference CUDA contexts do not load the candidate.
Only Q4 batch-four changes; Q6 and single-token decode retain existing dispatch.
The portable host implementation remains C. No production default is changed.

The initial model check passed numerically but failed admission because selection
was attempted at CUDA initialization, before the per-session switch. Its failed
metadata is retained. Selection now belongs to the resident plan. The corrected
31-token test passed both resets at maximum scaled error 5.24084294e-05. The
257-token ring test executed 568 Q4 integer kernel calls and passed both resets,
rewind/replacement and guards at 2.19546455e-05, under the unchanged 0.0005 bound.
run_model.py toggles the setting between reference and candidate session creation
and requires an actual nonzero integer kernel call count in new runs.

Both final C17 and C23 CUDA host libraries built with strict warning flags.
physical-removal-final-c17.log covers the final source without optional non-C
components: 82 tests completed, four expected ROCm/MLX hardware skips, no failures,
and the train/export/infer smoke lifecycle passed. Its tiny CPU timing is only a
smoke check. A separate CUDA-enabled removal suite (started from the initial
integration snapshot) and the 8K model test were still running at this checkpoint.
Do not treat their in-progress logs as successful validation. Full-model timing
and integrated sanitizer validation remain pending; no accepted score changes.

## Prepared generation comparison

build_generation.py produces a separate integer-generation-20261114 executable
and C17/C23 diagnostic core objects, preserving the earlier generation binary.
Only the existing diagnostic B512 prompt-ingestion override is applied; the
production library remains unchanged. The manifest hashes the CUDA host/source,
optional kernel, library, executable and harness inputs.

run_generation.py accepts an explicit integer switch, short/long context and
unique label. It requires completed 257/8192 lifecycle proofs with actual integer
dispatch, matching source hashes, the raw sanitizer proof, an intact build and
a conservative idle gate. The shared runner records the integer setting and uses
the continuous monitor with collector-death handling. Existing evidence is never
overwritten. A negative preflight while the 8K test was live rejected launch
before creating a run folder. Generation has not yet executed with this binary.
Run both setting orders and retain output parity and coverage before comparing;
the current kernel timings do not justify an accepted end-to-end speedup.

## Completed 8K lifecycle

The subsequent 8192-token ring-KV test exited zero. Both resets retained maximum
scaled error 1.63989979e-05 under the unchanged 0.0005 bound, including rewind and
replacement. The candidate trace records 568 actual Q4 integer kernel calls,
with aligned prefix, batched attention and output projection also exercised.
This is correctness evidence only; the concurrently running CUDA physical-removal
suite makes its wall time unsuitable for performance comparison. The suite was
still live at this checkpoint. No new accepted decode score is claimed.
