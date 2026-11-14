# Offline attention resource census

The script compiles the same six source files and NVRTC options used by the
cached-query raw correctness harness, targeting compute_120. It does not import
that harness (which creates a CUDA context), load the CUDA driver, or execute GPU
work. CUDA 13.1.115 ptxas assembles the generated PTX for SM120.

| Split entry | Baseline registers | Cached-query registers |
| --- | ---: | ---: |
| Linear | 56 | 60 |
| Ring | 48 | 48 |

All four entries report zero stack, spill stores and spill loads. The raw assembly
log also reports 16464 bytes shared memory for each. This is offline compiler
allocation, not driver-JIT resource use, achieved occupancy, runtime correctness,
or model throughput. The separate scoped C23 8K memcheck remains pending.

The initial parser read stderr only and refused with `Missing split kernels`:
this Windows ptxas writes its resource report to stdout. The corrected parser
reads both streams and requires all four split entries. No GPU run or performance
measurement occurred during either invocation. Result hashes cover the source,
compiler binaries, script, PTX, cubin and complete assembly output.
