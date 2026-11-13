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
