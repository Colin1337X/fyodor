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
