# Interleaved Q6 integer-product experiment

Original Fyodor, unintegrated, in `optcpp/matvec/interleaved_q6_integer.cu`.
No CMake, dispatch, production binary or running benchmark changes. The native
batch-two and batch-four interleaved implementations are copied with distinct
symbols and only Q6 instantiated. The sole arithmetic change uses the existing
`integer_scale_q6.cu` signed-zero-preserving coefficient construction:

```
float product = __int_as_float(__float_as_int((float)(sc*q)) |
                              ((sc ^ q) & (-2147483647-1)));
weight = d * product;
```

The signed scale is [-128,127], q is [-32,31], and product magnitude is at most
4096. The existing exhaustive C coefficient check in
`../decode-q6-integer-20261028/coefficient-check.json` covers 1,040,187,392 finite
half/scale/value combinations bit-exactly, including zero signs. Nonfinite
coefficients are excluded. That earlier scalar arithmetic evidence does not
prove GPU contraction or full-model correctness for these new batch kernels.
Input conversion/layout, weight bytes, accumulation and warp reduction order
are unchanged. The batch output projection and other Q6 matrices are the
motivation; earlier profiles identify them as a substantial decode cost.

`compile_only.py` verifies the exact source transformations and compiles native
and candidate symbols with NVRTC compute_120, C++11, fmad enabled and the existing
fast-math define. It requires both Q6 symbols and excludes Q4 candidate symbols.
Compilation passed, with source/tooling hashes and PTX hash retained. No CUDA
driver, context or kernel was used. This brief CPU compilation occurred during
the separate native generation diagnostic whose idle gate had already failed;
that run remains diagnostic, not an accepted score.

Pending: raw byte parity/guards for both batch widths over actual Q6 shapes and
finite coefficient/sign edge cases, real Compute Sanitizer validation, paired
kernel timing, and model-level parity/timing. Do not admit these kernels based
on compilation or the earlier coefficient proof. Q4's separate candidate and
its pending tests remain unchanged. No Q6 speedup is claimed.

## Prepared raw sweep

`check_kernel.py` prepares 712 cases: both batch widths over six shapes (three
small row tails and the three profiled Q6 shapes), plus all 350 combinations of
fourteen finite half patterns, five signed scales and five signed quantized
values at 256 columns/nine rows for each width. Scale/value sets include both
signs, zero and extrema. The large vocabulary matrix is 3840 by 262144.

Input interleaving is checked against a CPU transpose with guards. Existing
interleaved and candidate outputs must match native small-batch bytes and guards.
Small cases additionally retain the established float64 error bound of 2e-6
absolute-product sum plus 1e-6. The helper compiles both batch widths together;
source and harness hashes plus partial failure evidence are retained.

The script checks integrated sanitizer success and completed native generation
before its GPU-creating helper import. Syntax checking and the live-workload
negative preflight passed without creating an oracle. The sweep has NOT run.
`--memcheck` only selects the output filename; an actual Compute Sanitizer
invocation and terminal zero-error log are required separately.

## Completed raw GPU sweep

The subsequent 712-case sweep passed both batch widths, byte parity, guards and the small-case float64 bound. `oracle.json` retains the results and hashes. This supersedes the earlier pending raw-test status; sanitizer, timing and model-level checks remain pending. No Q6 speedup is claimed.

## Completed sanitizer and diagnostic timing

The actual Compute Sanitizer invocation passed all 712 cases with zero errors and unchanged source/harness hashes. Five offline validator tests passed. `memcheck-run.json` records the tool invocation, terminal exit and evidence hashes.

The repeated-weight probe requires raw correctness and sanitizer evidence before creating a GPU context. It runs both batch widths, both kernel orders and the three real Q6 shapes with interleaving included. Ten event samples per kernel/shape use 100 operations each after warmup. Outputs must remain byte-identical.

| Columns | Rows | Batch | Baseline / candidate speed ratio |
| --- | --- | --- | --- |

| 3840 | 2048 | 2 | 1.0409x |

| 3840 | 2048 | 4 | 1.1434x |

| 3840 | 262144 | 2 | 0.9876x |

| 3840 | 262144 | 4 | 1.0007x |

| 15360 | 3840 | 2 | 0.8529x |

| 15360 | 3840 | 4 | 1.0102x |


The batch-two 15360x3840 path regresses materially, and the large vocabulary
projection offers no consistent benefit. Do not replace general Q6 dispatch.
The batch-four 3840x2048 result merits a future shape-specific model experiment,
not automatic adoption. These synthetic repeated-weight measurements have no
continuous contamination audit and remain accepted_benchmark=false. They do not
change accepted llama.cpp scores or establish end-to-end decode improvement.
