# Bounded Q4 column-index experiment

Offline SM120 instruction inspection found 64-bit column induction/address
work in the fully unrolled metadata kernel. Partial unrolling reduced register
allocation but regressed all eight measured shapes. This candidate keeps full
unrolling and changes only column induction to unsigned 32-bit arithmetic.

It explicitly rejects column counts above 0xffffff00 or not divisible by 256;
this bounds the last increment and all group/lane offsets without wraparound.
Weight row offsets and strides remain 64-bit. Production integration would
need a matching host admission condition and fallback for unsupported sizes.
Floating-point expressions, group order, metadata decoding and warp reduction
remain unchanged. This is original Fyodor code under optional optcpp, with
no production dispatch. Compilation alone proves neither correctness nor speed.
Raw parity, boundary guards, sanitizers and multi-shape timing remain required.

NVRTC compilation and offline SM120 assembly passed with 66 registers and
zero stack, spill or shared-memory bytes. No GPU kernel has executed yet.

## Correct, but not promoted

All 207 raw parity cases passed byte-exact comparison with the serial and
current metadata kernels, output guards and the existing float64 small-shape
bound. Seven unsupported-input checks used null weight/input pointers and
required unchanged sentinel output: nonmultiples of 256, out-of-range 64-bit
column counts and batch two. All 242 raw input hashes independently verified.
Memcheck, racecheck and synccheck repeated all parity and rejection cases with
zero errors; 57 sanitizer input/evidence hash checks independently verified.

The first eight-shape probe was mixed. A longer confirmation increased warmup
from 30 to 1000 operations per order, used ABBA BAAB ordering and 40 event
observations per kernel/shape instead of ten. Both include activation packing
and compare timed outputs exactly. Positive values below mean slower:

| Columns | Rows | Initial latency change | Longer latency change |
| ---: | ---: | ---: | ---: |
| 3840 | 512 | +0.051% | +2.775% |
| 3840 | 2048 | +3.035% | +1.883% |
| 3840 | 4096 | -5.652% | -0.296% |
| 3840 | 8192 | -0.441% | +1.277% |
| 3840 | 15360 | -1.351% | +0.739% |
| 4096 | 3840 | +2.556% | -1.018% |
| 8192 | 3840 | +2.106% | -0.423% |
| 15360 | 3840 | +0.638% | -0.635% |

The largest expansion shape changed from a 1.351% apparent improvement to a
0.739% regression. The initial 5.652% improvement at 3840 columns/4096 rows
shrunk to 0.296%. Remaining differences are small and mixed. No consistent
benefit justifies model integration, especially with a new size constraint;
retain the existing kernel. These synthetic repeated weight-bank timings
have no continuous contamination acceptance and are not end-to-end scores.
No production dispatch changed and no accepted performance score increased.
