# Aligned byte-metadata Q4 row-dot diagnostic

Original Fyodor code, base2435df6, diagnostic only. Apply the already validated
160-byte Q4 layout to a generic-dimension row dot. Reference is the accepted
integer scale-product kernel, NOT the older original Q4 implementation. Legacy
JSON key nya_dot_12 is mapped to nya_dot_integer_12; fixed maps to nya_dot_byte_12.
Neither is a constant-dimension experiment. No upstream code copied.

Candidate retains original halves, byte scales/minima,12 zero padding bytes,
and128 payload bytes per block. Payload starts at32 and stride is160 rather
than144 bytes. Word-sized metadata loads extract bytes. Exact integer-product
arithmetic, F32 input, ordered accumulation and reduction remain unchanged.
Weight layout grows11.11%; both layouts coexist in diagnostics. Production C
packing tests are in decode-byte-runtime-20261028; these probes pack in Python
and test device arithmetic independently. No runtime dispatch/build changes.

## Correctness and timing

Nine shapes with random payload/metadata and fixed finite half coefficients,
nine more cycling14 finite half patterns (signed zero, subnormal, normal,
ordinary and maximum finite), and every bank of three rotating cases match
byte-for-byte, including guards. NaN/Inf coefficients, fused add/QKV and real-model
execution of this candidate are not validated here. The copied helper's Q6
branch is unchanged and not selected by the diagnostic wrapper.

RTX5060Ti SM120, driver617.42, NVRTC compute_120 C++11 fmad=true and
NYA_CUDA_FAST_MATH=1. Ten samples/variant, five each in opposite orders,
500 graph kernels/sample after warmups. All five monitored activities pass idle
and continuous competitor gates; no competitors killed, unrelated builds or GPU
tests during timing. CPU repacking/uploads finish before each measured case.
Packing and allocation costs are excluded from timing. Rotating cases use29,11,8
banks; raw weight_bank_bytes counts native bytes while summary.json includes
both footprints. Raw samples, sample standard deviations, source hashes and
kernel resources are retained (35 registers reference,37 candidate, zero locals).

| Case | Columns x rows | Reference ms | Candidate ms | Throughput gain |
|---|---:|---:|---:|---:|
| 12-hot | 256 x 9 | 0.000940 | 0.000899 | +4.54% |
| 12-hot | 768 x 65 | 0.001527 | 0.001430 | +6.75% |
| 12-hot | 2048 x 256 | 0.003036 | 0.002835 | +7.08% |
| 12-hot | 2048 x 2048 | 0.009255 | 0.007865 | +17.67% |
| 12-hot | 5632 x 2048 | 0.023825 | 0.019952 | +19.41% |
| 12-hot | 2048 x 32000 | 0.119484 | 0.108093 | +10.54% |
| 12-extended | 2304 x 257 | 0.003254 | 0.003047 | +6.82% |
| 12-extended | 4096 x 11008 | 0.078685 | 0.067205 | +17.08% |
| 12-extended | 8192 x 8192 | 0.124164 | 0.114015 | +8.90% |
| 12-rotating | 2048 x 2048 | 0.010332 | 0.009113 | +13.39% |
| 12-rotating | 5632 x 2048 | 0.026322 | 0.023567 | +11.69% |
| 12-rotating | 4096 x 11008 | 0.083037 | 0.075368 | +10.18% |

## Profile and decision

Fresh accepted-baseline profile (byte gate enabled) reports879.217ms matrix,
132.029ms norm,43.570ms rope,87.923ms attention,1.319ms gather. Q4 residual-add
at columns2048/5632 costs117.845/138.448ms and Q4 QKV63.696ms. This supports
investigating packed projections. Profiling is ungated, disables graphs and
includes submission gaps; it is not a matched performance acceptance result.

Proceed to optional dot/add/QKV runtime integration, reusing C-owned packed
weight handling while checking formats/strides at each dispatch. Preserve native
weights/fallbacks and Q6 V where applicable. Additional projection copies need
budget checks and honest memory accounting; kernel storage growth is not the
whole runtime overhead when native weights remain. Validate real-model CPU
parity, allocation/budget fallback, C17/C23 and physical removal before matched
decode acceptance. No change to current106.29% short /124.95% long llama.cpp
scores is claimed. >=125% on both remains unmet.

Reproduce run.py, run_rotating.py, run_scales.py serially into fresh output
directories. This checkpoint changes no production dispatch, model file format,
training behavior or portable build requirements.
