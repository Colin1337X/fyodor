# Decode FP16 scale-load investigation - October 15, 2026

Original Fyodor diagnostic. The previous goal turn rejected uniform dot-loop
unrolling. This experiment tests reduced scale-loading work in the same
profiled kernels, without editing production code. The prior short profile
places Q6_K dots at 17.6% of matrix time, remaining Q4_K dots at 42.5%, and
fused gate/up at about 40% before its accepted unrolling improvement.

## Variants

Q6_K blocks are 210 bytes, so they retain two-byte but not four-byte alignment.
A two-byte load is appropriate; a naive 32-bit metadata cast would be unsafe.
The standalone probe tests three alternatives to the committed baseline:

- `nya_dot_u2`: load each signed-scale byte pair as an unsigned short, extract
  the lane's byte and restore its signed value with `(value ^ 128) - 128`.
- `nya_dot_u4`: load the FP16 block multiplier directly as unsigned short,
  then apply the same existing `cvt.f32.f16` instruction.
- `nya_dot_u8`: combine both changes.

The names are inherited diagnostic identifiers, not loop-unroll factors.
Q4_K probing changes its two FP16 multiplier loads. The signed-scale-pair
change applies only to the Q6 branch, so Q4 variant u2 is a redundant control
and u8 duplicates u4. Gate/up probing uses keys 1 (baseline) and 2 (direct
FP16 loads), retaining the already accepted eight-way outer-loop unrolling.
No arithmetic sequence, quantization, launch geometry or activation dtype is
changed. Direct half loads require two-byte alignment; all allocated blocks
and tested row strides satisfy it. This is not a proposal to cast arbitrary
unaligned public inputs.

## Method and results

Compile the archived baseline and generated variants with production NVRTC
options. Compare every output byte and guard before timing. Q4 and Q6 each
cover six shapes; gate/up covers six including odd loop lengths and partial
row blocks. All candidate outputs match exactly. This is equivalence to the
production baseline, not a full independent mathematical oracle or sanitizer
pass. The fixed finite FP16 fields do not exhaustively test all half encodings.

Graphs contain 50 repeated kernels. Three warmup replays precede five samples
of ten replays, repeated in reverse variant order. All ten samples per cell
are retained. These hot-weight diagnostics include process snapshots but not
continuous monitoring or desktop idle gates, and cannot update the scorecard.

Throughput gain for direct FP16 scale loads alone (positive is faster):

| Kernel | Rows x columns | Gain |
| --- | --- | ---: |
| Q4_K dot | 9 x 256 | 1.77% |
| Q4_K dot | 65 x 768 | 4.82% |
| Q4_K dot | 256 x 2048 | 3.76% |
| Q4_K dot | 2048 x 2048 | 0.96% |
| Q4_K dot | 2048 x 5632 | 0.21% |
| Q4_K dot | 32000 x 2048 | 0.96% |
| Q6_K dot | 9 x 256 | 3.95% |
| Q6_K dot | 65 x 768 | 3.59% |
| Q6_K dot | 256 x 2048 | 1.13% |
| Q6_K dot | 2048 x 2048 | 2.89% |
| Q6_K dot | 2048 x 5632 | 3.01% |
| Q6_K dot | 32000 x 2048 | 1.46% |
| Q4_K gate/up | 9 x 256 | -0.68% |
| Q4_K gate/up | 5632 x 2048 | 1.39% |
| Q4_K gate/up | 11008 x 4096 | 4.59% |
| Q4_K gate/up | 65 x 768 | -4.78% |
| Q4_K gate/up | 257 x 2304 | -0.62% |
| Q4_K gate/up | 28672 x 8192 | 5.38% |

Reject signed-scale pair loading on its own: it regresses Q6 model-sized
shapes by about 1-7%. The combined variant is inconsistent. Direct FP16 loads
alone are the promising candidate: standalone results are modestly positive;
fused model-sized shapes improve, while some small shapes regress. No generic
helper replacement is accepted yet. The next runtime trial should scope the
aligned-load helper to these decode kernels, preserve the existing reference
math path, test alignment and FP16 edge encodings, then use full-model clean
before/after timing and regressions before retaining it.

Driver reports zero local-memory bytes for all variants; resource counts are
archived, but do not by themselves prove why timing differs. Accepted decode
remains **82.8% short / 103.5% long** against pinned llama.cpp. Production is
unchanged, and prefill/training optimization remains deferred.

## Evidence and reproduction

`probe.py 12 NEW_DIRECTORY` and `probe.py 14 NEW_DIRECTORY` run the standalone
cases. `probe_gate.py NEW_DIRECTORY` runs initial fused cases; append `extended`
for the larger/odd-length set. Output directories must not exist. Scripts use
bundled CUDA 13.1 NVRTC and the CUDA driver without PyTorch. All four diagnostic
processes exited zero. Results contain all samples, options and compiled source
hashes; companion files record kernel resources and process snapshots.
`baseline-matvec.cu.out` and `provenance.json` preserve the committed source.
Hardware: RTX 5060 Ti 16 GB, driver 617.42, compute_120, Windows.
