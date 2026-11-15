# Paired Q4 outer-loop unrolling: offline preparation

The measured large Q4_K expansion/down-projection is a dominant short-decode cost. This experiment preserves the existing paired-row kernel's expressions, accumulation order, reduction and layout while requesting outer-loop unrolling by factors 2 and 3. The hypothesis is that scheduling more block iterations together may hide latency or reduce loop overhead. No speedup has been measured or assumed.

All code is original Fyodor-derived optional CUDA C++ under /optcpp. No production dispatch or build source list changes. Deleting these optional files cannot introduce a dependency into the C core. This is not a llama.cpp source transplant; inspection of its Q4/Q8 dot path highlights a different activation-precision contract, which this experiment does not adopt.

## Offline findings (SM120, NVRTC/PTXAS 13.1.115)

| Kernel | Registers | Stack / spill store / spill load bytes | Static SASS instructions |
|---|---:|---:|---:|
| Baseline paired rows | 80 | 0 / 0 / 0 | 455 |
| Outer unroll 2 | 80 | 0 / 0 / 0 | 1527 |
| Outer unroll 3 | 80 | 8 / 8 / 8 | 2127 |

Factor 2 retains the baseline resource allocation but increases code footprint substantially. Factor 3 incurs local-memory spilling and is deprioritized. Static instruction counts include compiler-generated control/tail paths and are not dynamic work counts, occupancy measurements, correctness evidence or throughput estimates. The larger instruction footprint may outweigh any scheduling benefit.

compile.py, assemble.py and inspect.py complete without loading a CUDA driver or creating a GPU context. Retained manifests hash exact inputs, tools and outputs. common.py and check_kernel.py are prepared for factor-2 GPU validation but have NOT run. They preserve the original float64 bounds, require byte-exact equivalence to the native metadata kernel, test output/input guards and unsupported-shape rejection, and reject another GPU workload before creating a context. Factor 3 has no numerical or GPU timing evidence.

The existing 8K compiler-instrumented model check is running separately and must finish before these GPU tests. Do not import common.py during that run. No candidate timing, sanitizer pass, real-model win, or integration is claimed. Accepted decode remains 85.61025294% short / 106.8707255% long of llama.cpp; target 125%+ both.
